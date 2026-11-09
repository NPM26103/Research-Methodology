"""A single evaluator for primary test and exploratory validation."""

import importlib.metadata, json, platform, sys, traceback
import torch
from rsm.common import load_partition, device, seed_all
from rsm.model import Recognizer
from .io import new_attempt, latest_completed, freeze_checkpoint, now, source_hashes
from .safety import sha, save_json, snapshot, changes
from .audit import audit_data
from .diagnostics import state_sha, sanity_checks, evaluate_conditions, summarize


def reuse(run, name, digest):
    """Verify hashes before reusing a previously completed evaluation."""
    out = latest_completed(run, name)
    if out is None:
        return None
    complete = json.loads((out / "completion.json").read_text())
    if complete["checkpoint_sha256"] != digest:
        raise ValueError("Completed output belongs to another checkpoint")
    for path, expected in complete["output_sha256"].items():
        if sha(out / path) != expected:
            raise ValueError("Completed output changed: " + path)
    print("Reusing completed output:", out, flush=True)
    return out


def finish(out, digest, extra):
    """Record hashes only after every evaluation output has been written successfully."""
    files = {
        p.relative_to(out).as_posix(): sha(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    save_json(
        out / "completion.json",
        {
            "complete": True,
            "finished_utc": now(),
            "checkpoint_sha256": digest,
            "output_sha256": files,
            **extra,
        },
    )


def preflight(run, processed, dev_name):
    """Run validation sanity checks before the first main test evaluation."""
    checkpoint, digest = freeze_checkpoint(run)
    previous = reuse(run, "validation_sanity", digest)
    if previous:
        return previous
    return work(
        run,
        processed,
        dev_name,
        "sanity",
        new_attempt(run, "validation_sanity"),
        checkpoint,
        digest,
    )


def evaluate(run, processed, dev_name, partition, study):
    """Enforce main-test-before-exploration ordering and reuse completed results."""
    checkpoint, digest = freeze_checkpoint(run)
    needed = "validation_sanity" if partition == "test" else "main_test"
    if not reuse(run, needed, digest):
        raise ValueError("Complete " + needed + " first")
    name = "main_test" if partition == "test" else "exploratory_validation"
    previous = reuse(run, name, digest)
    if previous:
        return previous
    return work(
        run,
        processed,
        dev_name,
        partition,
        new_attempt(run, name),
        checkpoint,
        digest,
        study,
    )


def work(run, processed, dev_name, part, out, checkpoint, digest, study=None):
    """Evaluate one locked checkpoint while verifying protected input bytes stay unchanged."""
    torch.set_num_threads(2)
    seed_all(26)
    dev = device(dev_name)
    roots = {"checkpoint": checkpoint, "processed": processed}
    before = snapshot(roots)
    save_json(out / "input_snapshot_before.json", before)
    error = None
    try:
        versions = {
            k: importlib.metadata.version(k)
            for k in [
                "torch",
                "torchaudio",
                "torchvision",
                "timm",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
            ]
        }
        save_json(
            out / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "versions": versions,
                "device": str(dev),
                "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else None,
            },
        )
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = Recognizer(ck["config"], len(ck["label_map"])).to(dev)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before_state = state_sha(model)
        audit_data(processed, ck, out)
        if part == "sanity":
            X, y, ids, _, _ = load_partition(processed, "val", ck["data_hashes"])
            T, _, _, _, _ = load_partition(processed, "train", ck["data_hashes"])
            sanity_checks(model, ck, X, y, T, ids, dev, out, ck.get("val_correct"))
        else:
            X, y, ids, mapping, hashes = load_partition(
                processed, part, ck["data_hashes"]
            )
            if part == "test" and set(ids) & (
                set(ck["train_ids"]) | set(ck["val_ids"])
            ):
                raise ValueError("Test overlaps train/val")
            snrs = study["test_snrs"] if part == "test" else study["validation_snrs"]
            replicates = study["gaussian_replicates"]
            role = (
                "main_test_experiment"
                if part == "test"
                else "exploratory_validation_diagnostics"
            )
            save_json(
                out / "evaluation_metadata.json",
                {
                    "partition": part,
                    "analysis_role": role,
                    "checkpoint_sha256": digest,
                    "epoch": int(ck["epoch"]),
                    "config": ck["config"],
                    "data_hashes": hashes,
                    "sample_ids": ids,
                    "n": len(y),
                    "num_classes": len(mapping),
                    "snrs": snrs,
                    "master_seed": study["master_seed"],
                    "gaussian_replicates": replicates,
                    "primary_snr_db": 20,
                    "source_sha256": source_hashes(),
                },
            )
            df = evaluate_conditions(
                model,
                X,
                y,
                ids,
                mapping,
                digest,
                dev,
                out,
                snrs,
                replicates,
                part,
                master_seed=study["master_seed"],
            )
            summary = summarize(
                df,
                mapping,
                snrs,
                replicates,
                out,
                study["bootstrap_repetitions"],
                part,
                master_seed=study["master_seed"],
            )
            if part == "val":
                clean = df[df.condition == "clean"]
                correct = int((clean.pred == clean.true_label).sum())
                if ck.get("val_correct") is not None and correct != ck["val_correct"]:
                    raise ValueError("Validation differs from saved best checkpoint")
            print(
                summary[["condition", "target_snr", "accuracy", "macro_f1"]].to_string(
                    index=False
                ),
                flush=True,
            )
        if state_sha(model) != before_state or any(m.training for m in model.modules()):
            raise ValueError("Model state or eval mode changed")
    except BaseException as exc:
        error = exc
        save_json(
            out / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        after = snapshot(roots)
        save_json(out / "input_snapshot_after.json", after)
        modified = changes(before, after)
        save_json(
            out / "preservation_check.json",
            {"passed": not modified, "changed_files": modified},
        )
        if modified and error is None:
            error = RuntimeError("Input bytes changed during evaluation")
    if error is not None:
        raise error
    finish(
        out,
        digest,
        {
            "partition": "val" if part == "sanity" else part,
            "purpose": "sanity" if part == "sanity" else role,
            "model_state_unchanged": True,
            "protected_inputs_unchanged": True,
            "prediction_rows": 0 if part == "sanity" else len(df),
        },
    )
    print("COMPLETE:", out, flush=True)
    return out
