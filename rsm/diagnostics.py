"""Inference-only validation checks, preserving the original waveform frontend."""

import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from sklearn.metrics import f1_score, confusion_matrix
from rsm.model import Frontend
from rsm.noise import at_snr, fgsm_direction
from rsm.metrics import analyze, write_report_tables
from rsm.seeding import derived_seed
from .safety import save_json


def state_sha(model):
    """Fingerprint model parameters and buffers to detect changes during evaluation."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sanity_checks(model, ck, X, y, T, ids, dev, out, expected_correct):
    """Verify the selected baseline, zero-noise behavior, training statistics and gradients."""
    if any(m.training for m in model.modules()):
        raise ValueError("All modules must be in eval mode")
    probe = Frontend(ck["config"]["frontend"]).to(dev).eval()
    probe.fit(T, dev)  # separate temporary frontend, never the saved model
    for name in ["mean", "std"]:
        if not torch.allclose(
            getattr(probe, name), getattr(model.frontend, name), atol=1e-4, rtol=1e-4
        ):
            raise ValueError("Train-only normalization statistics mismatch: " + name)
    with torch.no_grad():
        clean = torch.cat([model(b.to(dev)).cpu() for b in X.split(16)])
        zero = torch.cat(
            [
                model((b + torch.zeros_like(b)).clamp(-1, 1).to(dev)).cpu()
                for b in X.split(16)
            ]
        )
    if clean.shape != (len(y), len(ck["label_map"])) or not torch.isfinite(clean).all():
        raise ValueError("Invalid clean logits shape/values")
    if not torch.allclose(clean, zero, atol=1e-6, rtol=1e-5) or not torch.equal(
        clean.argmax(1), zero.argmax(1)
    ):
        raise ValueError("Zero-noise output differs from clean")
    correct = int((clean.argmax(1) == y).sum())
    macro_f1 = float(
        f1_score(
            y.numpy(),
            clean.argmax(1).numpy(),
            labels=np.arange(clean.shape[1]),
            average="macro",
            zero_division=0,
        )
    )
    result = {
        "partition": "val",
        "n": len(y),
        "logits_shape": list(clean.shape),
        "clean_correct": correct,
        "expected_correct": expected_correct,
        "accuracy": correct / len(y),
        "macro_f1": macro_f1,
        "saved_val_macro_f1": float(ck["val_macro_f1"]),
        "zero_noise_max_logit_error": float((clean - zero).abs().max()),
        "train_statistics_match": True,
        "all_modules_in_eval": True,
        "frontend_mean_saved": float(model.frontend.mean),
        "frontend_std_saved": float(model.frontend.std),
        "frontend_mean_recomputed": float(probe.mean),
        "frontend_std_recomputed": float(probe.std),
    }
    result["baseline_reproduced"] = (
        expected_correct is None or correct == expected_correct
    ) and bool(np.isclose(macro_f1, ck["val_macro_f1"], atol=1e-8, rtol=1e-8))
    save_json(out / "validation_checks.json", result)
    if not result["baseline_reproduced"]:
        raise ValueError(
            "Clean validation differs from saved baseline; inspect environment/checkpoint before proceeding"
        )
    rows = []
    for label in range(clean.shape[1]):
        i = int(torch.where(y == label)[0][0])
        x = X[i : i + 1].to(dev).clone().requires_grad_(True)
        yy = y[i : i + 1].to(dev)
        loss = F.cross_entropy(model(x), yy)
        g = torch.autograd.grad(loss, x)[0]
        if (
            not torch.isfinite(loss)
            or not torch.isfinite(g).all()
            or float(g.abs().max()) == 0
        ):
            raise ValueError("Nonfinite/zero waveform gradient: " + ids[i])
        direction = fgsm_direction(model, x.detach(), yy)
        if not torch.equal(direction, g.sign()):
            raise ValueError("FGSM sign differs from independent gradient")
        for relative in [1e-6, 1e-5, 1e-4]:
            step = x.detach().square().mean().sqrt() * relative
            with torch.no_grad():
                plus = F.cross_entropy(
                    model((x.detach() + step * direction).clamp(-1, 1)), yy
                )
                minus = F.cross_entropy(
                    model((x.detach() - step * direction).clamp(-1, 1)), yy
                )
            if not torch.isfinite(plus) or not torch.isfinite(minus):
                raise ValueError("Nonfinite perturbed loss")
            rows.append(
                {
                    "sample_id": ids[i],
                    "true_label": label,
                    "relative_rms_step": relative,
                    "gradient_abs_max": float(g.abs().max()),
                    "gradient_nonzero_fraction": float((g != 0).float().mean()),
                    "positive_direction_derivative": float((g * direction).sum()),
                    "clean_ce": float(loss.detach()),
                    "plus_ce": float(plus),
                    "minus_ce": float(minus),
                    "delta_ce": float(plus - loss.detach()),
                    "central_difference": float((plus - minus) / (2 * step)),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "gradient_checks.csv", index=False)
    save_json(
        out / "gradient_summary.json",
        {
            "finite_nonzero_gradients_and_fgsm_sign_passed": True,
            "samples": len(ck["label_map"]),
            "decreasing_loss_probes": int((table.delta_ce < -1e-7).sum()),
            "zero_change_probes": int((table.delta_ce == 0).sum()),
            "interpretation": "Finite differences are diagnostic, not a pass criterion: tiny steps may round to zero; larger steps may cross nonlinear boundaries. Investigate sustained decreases.",
        },
    )
    return clean.argmax(1), result


def evaluate_conditions(
    model,
    X,
    y,
    ids,
    mapping,
    checkpoint_hash,
    dev,
    out,
    snrs,
    replicates,
    partition,
    master_seed=26,
):
    """Evaluate paired conditions using master-seed-derived Gaussian streams and SNR audits."""
    captured = {}

    def hook(_module, _args, output):
        captured["feature"] = output.detach().cpu()

    handle = model.frontend.register_forward_hook(hook)
    rows, examples = [], {}
    inverse = {v: k for k, v in mapping.items()}
    # A priori fixed examples, not selected for attack success.
    example_labels = [mapping[k] for k in ["0", "l", "p"] if k in mapping]
    if not example_labels:
        example_labels = list(mapping.values())[:3]
    chosen = {int(torch.where(y == label)[0][0]) for label in example_labels}
    audio_dir = out / "examples"
    audio_dir.mkdir()

    def predict(xp, label):
        with torch.no_grad():
            logits = model(xp)
            if logits.shape != (1, len(mapping)) or not torch.isfinite(logits).all():
                raise ValueError("Invalid inference logits")
            prob = logits.softmax(1)
            return {
                "pred": int(logits.argmax(1)),
                "confidence": float(prob.max()),
                "true_class_probability": float(prob[0, label]),
                "cross_entropy": float(
                    F.cross_entropy(logits, torch.tensor([label], device=dev))
                ),
            }, captured["feature"].clone()

    try:
        for i, sid in enumerate(ids):
            x = X[i : i + 1].to(dev)
            yy = y[i : i + 1].to(dev)
            label = int(yy.item())
            clean_result, clean_feature = predict(x, label)
            base = {
                "checkpoint_id": checkpoint_hash,
                "sample_id": sid,
                "true_label": label,
                "master_seed": master_seed,
            }
            clean_row = dict(
                base,
                condition="clean",
                target_snr=np.nan,
                replicate_id=-1,
                actual_snr=np.nan,
                independent_snr_db=np.nan,
                snr_recompute_error_db=0.0,
                clip_fraction=0.0,
                scale=0.0,
                feature_rmse_in_train_std=0.0,
                feature_mean=float(clean_feature.mean()),
                feature_std=float(clean_feature.std(unbiased=False)),
                **clean_result,
            )
            rows.append(clean_row)
            if i in chosen:
                examples[sid] = [
                    (
                        "Clean",
                        x.cpu().numpy()[0],
                        clean_feature.numpy()[0, 0],
                        clean_result,
                    )
                ]
                wavfile.write(
                    audio_dir / f"{sid}_clean.wav",
                    44100,
                    x.cpu().numpy()[0].astype(np.float32),
                )

            def append_noisy(condition, snr, replicate_id, direction):
                xp, audit = at_snr(x, direction, float(snr))
                # Independent NumPy recomputation uses the exact float32 waveform
                # sent to the model, after clipping and dtype conversion.
                a, b = x.detach().cpu().numpy().astype(
                    np.float64
                ), xp.detach().cpu().numpy().astype(np.float64)
                measured = 10 * math.log10(
                    float(np.mean(a * a) / np.mean((b - a) ** 2))
                )
                reported = float(audit["actual_snr"].item())
                if (
                    not math.isfinite(measured)
                    or abs(measured - snr) > 0.1
                    or abs(measured - reported) > 1e-5
                ):
                    raise ValueError("Independent SNR check failed: " + sid)
                result, feature = predict(xp, label)
                rows.append(
                    dict(
                        base,
                        condition=condition,
                        target_snr=snr,
                        replicate_id=replicate_id,
                        independent_snr_db=measured,
                        snr_recompute_error_db=abs(measured - reported),
                        feature_rmse_in_train_std=float(
                            (feature - clean_feature).square().mean().sqrt()
                        ),
                        feature_mean=float(feature.mean()),
                        feature_std=float(feature.std(unbiased=False)),
                        **result,
                        **{k: float(v.item()) for k, v in audit.items()},
                    )
                )
                if (
                    i in chosen
                    and snr in [40, 30, 20]
                    and (condition == "fgsm" or replicate_id == replicates[0])
                ):
                    name = f"{condition}_{snr}dB"
                    wavfile.write(
                        audio_dir / f"{sid}_{name}.wav",
                        44100,
                        xp.detach().cpu().numpy()[0].astype(np.float32),
                    )
                    examples[sid].append(
                        (f"{condition} {snr} dB", b[0], feature.numpy()[0, 0], result)
                    )

            direction = fgsm_direction(model, x, yy)
            for snr in snrs:
                append_noisy("fgsm", snr, -1, direction)
            for replicate_id in replicates:
                gen = torch.Generator(device="cpu").manual_seed(
                    derived_seed(master_seed, "gaussian", replicate_id, sid)
                )
                u = torch.randn(x.shape, generator=gen, dtype=torch.float32).to(dev)
                for snr in snrs:
                    append_noisy("gaussian", snr, replicate_id, u)
            if (i + 1) % 10 == 0 or i + 1 == len(ids):
                print(f"{partition}: {i+1}/{len(ids)} samples", flush=True)
    finally:
        handle.remove()
    df = pd.DataFrame(rows)
    expected = len(ids) * (1 + len(snrs) * (1 + len(replicates)))
    if len(df) != expected:
        raise ValueError("Unexpected prediction row count")
    df.to_csv(out / "predictions.csv", index=False)
    save_json(
        out / "snr_audit.json",
        {
            "noisy_rows": int(df.condition.ne("clean").sum()),
            "max_target_error_db": float(
                (df.independent_snr_db - df.target_snr).abs().max()
            ),
            "max_recomputation_difference_db": float(df.snr_recompute_error_db.max()),
            "max_clip_fraction": float(df.clip_fraction.max()),
            "tolerance_db": 0.1,
        },
    )
    render_examples(examples, inverse, audio_dir)
    return df


def render_examples(examples, inverse, out):
    """Use shared amplitude/color scales for a priori selected clean/noisy examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sid, items in examples.items():
        lo = min(z.min() for _, _, z, _ in items)
        hi = max(z.max() for _, _, z, _ in items)
        limit = max(np.abs(x).max() for _, x, _, _ in items) * 1.05
        fig, axes = plt.subplots(
            len(items),
            2,
            figsize=(11, 2.2 * len(items)),
            layout="constrained",
            squeeze=False,
        )
        for axis, (name, x, z, result) in zip(axes, items):
            axis[0].plot(np.arange(len(x)) / 44100, x, lw=0.5, color="#316c9c")
            axis[0].set(
                ylim=(-limit, limit), ylabel="Amplitude", xlabel="Time (s)", title=name
            )
            im = axis[1].imshow(
                z, origin="lower", aspect="auto", cmap="magma", vmin=lo, vmax=hi
            )
            axis[1].set(
                xlabel="Time bin",
                ylabel="Mel bin",
                title=f"Pred: {inverse[result['pred']]} | confidence {result['confidence']:.3f}",
            )
        fig.suptitle(sid + " — shared waveform and feature scales")
        fig.colorbar(
            im, ax=axes[:, 1].tolist(), label="Train-standardized log-mel", shrink=0.6
        )
        fig.savefig(out / f"{sid}_comparison.png", dpi=140)
        plt.close(fig)
    (out / "README.txt").write_text(
        "WAV files are float32 at 44100 Hz, with the exact inference amplitudes. Do not independently normalize playback clips when comparing. Use IPython Audio(path, normalize=False). Images share amplitude/color scales within each sample. Examples were fixed before seeing outcomes.\n"
    )


def summarize(df, mapping, snrs, replicates, out, reps, partition, master_seed=26):
    """Save measured metrics, paired effects, diagnostic tables and report figures."""
    per, summary, effects, clean, groups = analyze(
        df, len(mapping), snrs, replicates, reps=reps, master_seed=master_seed
    )
    role = (
        "exploratory_validation_diagnostics"
        if partition == "val"
        else "main_test_experiment"
    )
    if partition == "val":
        effects = effects.rename(columns={"primary": "reference_to_main_test_snr"})
    effects["analysis_role"] = role
    for name, table in [
        ("metrics_per_replicate", per),
        ("metrics_summary", summary),
        ("paired_effects", effects),
    ]:
        table.to_csv(out / (name + ".csv"), index=False)
    inverse = {v: k for k, v in mapping.items()}
    concentration, distribution = [], []
    for (cond, snr, replicate_id), g in df.groupby(
        ["condition", "target_snr", "replicate_id"], dropna=False
    ):
        counts = g.pred.value_counts()
        p = counts.to_numpy() / len(g)
        concentration.append(
            {
                "condition": cond,
                "target_snr": snr,
                "replicate_id": replicate_id,
                "unique_predicted_classes": len(counts),
                "most_predicted_key": inverse[int(counts.index[0])],
                "largest_class_fraction": float(counts.iloc[0] / len(g)),
                "predicted_label_entropy_bits": float(-(p * np.log2(p)).sum()),
                "mean_confidence": float(g.confidence.mean()),
                "mean_cross_entropy": float(g.cross_entropy.mean()),
                "feature_rmse_in_train_std": float(
                    np.sqrt(np.mean(g.feature_rmse_in_train_std**2))
                ),
            }
        )
        for label in range(len(mapping)):
            distribution.append(
                {
                    "condition": cond,
                    "target_snr": snr,
                    "replicate_id": replicate_id,
                    "predicted_key": inverse[label],
                    "count": int(counts.get(label, 0)),
                    "fraction": float(counts.get(label, 0) / len(g)),
                }
            )
    pd.DataFrame(concentration).to_csv(
        out / "prediction_concentration.csv", index=False
    )
    pd.DataFrame(distribution).to_csv(out / "prediction_distribution.csv", index=False)
    df.groupby(["condition", "target_snr", "replicate_id"], dropna=False).agg(
        mean_normalized=("feature_mean", "mean"),
        mean_within_clip_std=("feature_std", "mean"),
        feature_rmse_in_train_std=(
            "feature_rmse_in_train_std",
            lambda v: np.sqrt(np.mean(v * v)),
        ),
    ).reset_index().to_csv(out / "frontend_shift.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cond, color in [("gaussian", "#2f6ea4"), ("fgsm", "#b34545")]:
        q = summary[summary.condition == cond].sort_values("target_snr")
        ax.errorbar(
            q.target_snr,
            100 * q.accuracy,
            yerr=100 * q.accuracy_std.fillna(0),
            marker="o",
            capsize=3,
            label=cond,
            color=color,
        )
    ca = float(summary.loc[summary.condition == "clean", "accuracy"].iloc[0])
    ax.axhline(100 * ca, color="#444444", ls="--", label=f"clean {partition}")
    ax.axhline(
        100 / len(mapping), color="#999999", ls=":", label="1 / number of classes"
    )
    ax.set(
        xticks=sorted(snrs),
        ylim=(0, 102),
        xlabel="SNR (dB; larger = less noise)",
        ylabel="Accuracy (%)",
        title=f"{partition}: fixed checkpoint, n={len(clean)} clips",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_snr.png", dpi=180)
    fig.savefig(out / "accuracy_vs_snr.pdf")
    plt.close(fig)
    candidates = [("clean", clean.true_label.to_numpy(), clean.pred.to_numpy())]
    if ("fgsm", 20.0, -1) in groups:
        candidates.append(
            ("fgsm_20", clean.true_label.to_numpy(), groups[("fgsm", 20.0, -1)])
        )
        candidates.append(
            (
                "gaussian_20_pooled",
                np.tile(clean.true_label.to_numpy(), len(replicates)),
                np.concatenate([groups[("gaussian", 20.0, s)] for s in replicates]),
            )
        )
    for name, yy, pp in candidates:
        cm = confusion_matrix(yy, pp, labels=np.arange(len(mapping)))
        pd.DataFrame(
            cm,
            index=[inverse[i] for i in range(len(mapping))],
            columns=[inverse[i] for i in range(len(mapping))],
        ).to_csv(out / f"confusion_{name}.csv")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            cm / np.maximum(cm.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues"
        )
        ax.set(
            xticks=range(len(mapping)),
            yticks=range(len(mapping)),
            xticklabels=[inverse[i] for i in range(len(mapping))],
            yticklabels=[inverse[i] for i in range(len(mapping))],
            xlabel="Predicted key",
            ylabel="True key",
            title=f"{partition}: {name}",
        )
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, label="Row proportion")
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name}.png", dpi=170)
        fig.savefig(out / f"confusion_{name}.pdf")
        plt.close(fig)
    save_json(
        out / "analysis_metadata.json",
        {
            "partition": partition,
            "analysis_role": role,
            "bootstrap_repetitions": reps,
            "bootstrap_seed": master_seed,
            "master_seed": master_seed,
            "stratified_by": "true class; paired by sample",
            "ci_scope": "Fixed checkpoint, recordings and Gaussian replicate streams; does not cover training-seed/session variability.",
            "main_test_primary_snr_db": 20,
            "note": "Validation diagnostics are exploratory and cannot replace the primary test comparison. Disclose prior inspection of this dataset; reconstructing a split does not create a new unseen dataset. Uniform-guess accuracy does not imply uniform predictions.",
        },
    )
    write_report_tables(summary, effects, out, partition)
    return summary
