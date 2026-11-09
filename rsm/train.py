"""Train/validate only. The final test partition is not read by this command."""

import argparse, json, platform, subprocess, time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from .common import load_partition, seed_all, device
from .model import Recognizer
from .seeding import derived_seed


@torch.no_grad()
def validate(model, X, y, dev, batch_size, nclasses):
    """Evaluate clean validation data without augmentation or parameter updates."""
    model.eval()
    loss = 0.0
    pred = []
    for i in range(0, len(y), batch_size):
        z = model(X[i : i + batch_size].to(dev))
        yb = y[i : i + batch_size].to(dev)
        loss += float(torch.nn.functional.cross_entropy(z, yb, reduction="sum").item())
        pred.extend(z.argmax(1).cpu().tolist())
    yy = y.numpy()
    pred = np.asarray(pred)
    return (
        loss / len(y),
        float((pred == yy).mean()),
        float(
            f1_score(
                yy, pred, labels=list(range(nclasses)), average="macro", zero_division=0
            )
        ),
    )


def plots(history, out):
    """Export loss, accuracy, validation F1 and the actual per-epoch learning rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.DataFrame(history)
    best = table.sort_values(
        ["val_macro_f1", "val_loss"], ascending=[False, True]
    ).iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(table.epoch, table.train_loss, label="Train (augmented)")
    axes[0, 0].plot(table.epoch, table.val_loss, label="Validation")
    axes[0, 0].set(ylabel="Cross-entropy loss")
    axes[0, 1].plot(table.epoch, table.train_accuracy, label="Train (augmented)")
    axes[0, 1].plot(table.epoch, table.val_accuracy, label="Validation")
    axes[0, 1].set(ylabel="Accuracy", ylim=(0, 1))
    axes[1, 0].plot(table.epoch, table.val_macro_f1, label="Validation macro-F1")
    axes[1, 0].set(ylabel="Macro-F1", ylim=(0, 1))
    axes[1, 1].plot(table.epoch, table.learning_rate, label="Learning rate used")
    axes[1, 1].set(ylabel="Learning rate", yscale="log")
    for axis in axes.flat:
        axis.axvline(
            best.epoch,
            color="grey",
            linestyle=":",
            label=f"Best epoch: {int(best.epoch)}",
        )
        axis.set(xlabel="Epoch")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Training history — checkpoint selection uses validation only")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=180)
    fig.savefig(out / "training_curves.pdf")
    plt.close(fig)


def main():
    """Train on train only, select on validation, and save resumable epoch checkpoints."""
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--config", default="configs/run.json")
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--model", choices=["coatnet_0_224", "tiny_cnn"])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument(
        "--session-epochs",
        type=int,
        help="Pause after N epochs in this session, without changing the training schedule",
    )
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    dev = device(a.device)
    out = Path(a.run)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    for cli, k in [
        (a.epochs, "epochs"),
        (a.model, "model"),
        (a.batch_size, "batch_size"),
    ]:
        if cli is not None:
            cfg[k] = cli
    if cfg["model"] == "tiny_cnn":
        cfg["baseline_description"] = (
            "Alternative small CNN, intended for pipeline smoke tests"
        )
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("epochs and batch size must be positive")
    if (out / "last.pt").exists() and not a.resume:
        raise ValueError("Run exists; use a new --run or --resume")
    # initialize model/data using the common master seed.
    seed_all(cfg["seed"])
    X, y, ids, mapping, hashes = load_partition(a.processed, "train")
    V, vy, vids, _, _ = load_partition(a.processed, "val", hashes)
    model = Recognizer(cfg, len(mapping)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.002, total_iters=cfg["epochs"]
    )
    history = []
    start = 0
    best_f1 = -1.0
    best_loss = float("inf")
    stale = 0
    best_epoch = 0
    # restore optimizer/scheduler, or fit normalization on train only.
    if a.resume:
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if ck["config"] != cfg or ck["data_hashes"] != hashes:
            raise ValueError("Resume config/data differ")
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start = ck["epoch"]
        history = ck["history"]
        best_f1 = ck["best_f1"]
        best_loss = ck["best_loss"]
        stale = ck["stale"]
        best_epoch = ck["best_epoch"]
    else:
        model.frontend.fit(X, dev)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(dev),
        "torch": str(torch.__version__),
        "packages": {
            k: version(k)
            for k in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "timm",
                "torchvision",
                "torchaudio",
            ]
        },
    }
    if dev.type == "cuda":
        environment["gpu"] = torch.cuda.get_device_name(dev)
    if (out / "environment.json").exists():
        previous = json.loads((out / "environment.json").read_text())
        if (
            previous["torch"] != environment["torch"]
            or previous["packages"] != environment["packages"]
        ):
            raise ValueError(
                "Package versions changed; restore the saved training environment before resuming"
            )
    else:
        (out / "environment.json").write_text(json.dumps(environment, indent=2))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    frozen = subprocess.run(
        [__import__("sys").executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    )
    if frozen.returncode:
        raise RuntimeError("Could not record pip freeze")
    if not (out / "requirements-lock.txt").exists():
        (out / "requirements-lock.txt").write_text(frozen.stdout)
    print(
        "Device:",
        dev,
        "model:",
        cfg["model"],
        "parameters:",
        sum(p.numel() for p in model.parameters()),
        "train:",
        len(y),
        "val:",
        len(vy),
        flush=True,
    )
    base = {
        "config": cfg,
        "label_map": mapping,
        "data_hashes": hashes,
        "train_ids": ids,
        "val_ids": vids,
        "environment": environment,
    }
    dataset = torch.utils.data.TensorDataset(X, y)
    if a.resume and cfg["patience"] > 0 and stale >= cfg["patience"]:
        plots(history, out)
        (out / "training_complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "reason": "early_stopping",
                    "last_epoch": start,
                    "best_epoch": best_epoch,
                },
                indent=2,
            )
        )
        print("This run already reached early stopping; keeping its best checkpoint.")
        return
    if a.session_epochs is not None and a.session_epochs < 1:
        raise ValueError("session-epochs must be positive")
    session_end = (
        cfg["epochs"]
        if a.session_epochs is None
        else min(cfg["epochs"], start + a.session_epochs)
    )
    # deterministic epoch streams make epoch-boundary resume repeatable.
    for epoch in range(start, session_end):
        epoch_seed = derived_seed(cfg["seed"], "training_epoch", epoch)
        seed_all(epoch_seed)
        learning_rate = opt.param_groups[0]["lr"]
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(epoch_seed),
        )
        model.train()
        loss_sum = 0.0
        correct = 0
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
        vl, va, vf = validate(model, V, vy, dev, cfg["batch_size"], len(mapping))
        scheduler.step()
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=loss_sum / len(y),
                train_accuracy=correct / len(y),
                val_loss=vl,
                val_accuracy=va,
                val_macro_f1=vf,
                learning_rate=learning_rate,
                seconds=time.perf_counter() - t0,
            )
        )
        # checkpoint selection uses validation macro-F1, then validation loss.
        if vf > best_f1 or (vf == best_f1 and vl < best_loss):
            best_f1 = vf
            best_loss = vl
            stale = 0
            best_epoch = epoch + 1
            best_temp = out / "best.tmp"
            torch.save(
                dict(
                    base,
                    model_state=model.state_dict(),
                    epoch=epoch + 1,
                    val_macro_f1=vf,
                    val_loss=vl,
                    val_accuracy=va,
                    val_correct=int(round(va * len(vy))),
                ),
                best_temp,
            )
            best_temp.replace(out / "best.pt")
        else:
            stale += 1
        # persist optimizer and scheduler after each complete epoch.
        checkpoint = dict(
            base,
            model_state=model.state_dict(),
            optimizer_state=opt.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=epoch + 1,
            history=history,
            best_f1=best_f1,
            best_loss=best_loss,
            stale=stale,
            best_epoch=best_epoch,
        )
        temp = out / "last.tmp"
        torch.save(checkpoint, temp)
        temp.replace(out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(
            f"epoch {epoch+1}: train loss={loss_sum/len(y):.4f}, val acc={va:.4f}, val F1={vf:.4f}, seconds={history[-1]['seconds']:.1f}",
            flush=True,
        )
        if cfg["patience"] > 0 and stale >= cfg["patience"]:
            print("Early stopping on validation")
            break
    if history:
        plots(history, out)
    last_epoch = history[-1]["epoch"] if history else start
    complete = last_epoch >= cfg["epochs"] or (
        cfg["patience"] > 0 and stale >= cfg["patience"]
    )
    if complete:
        (out / "training_complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "reason": (
                        "early_stopping" if last_epoch < cfg["epochs"] else "max_epochs"
                    ),
                    "last_epoch": last_epoch,
                    "best_epoch": best_epoch,
                },
                indent=2,
            )
        )
    print(
        f"Best validation epoch={best_epoch}, macro-F1={best_f1:.4f}. Training complete={complete}. Test has not been evaluated."
    )


if __name__ == "__main__":
    main()
