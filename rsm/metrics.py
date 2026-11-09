"""Validate paired predictions and produce report tables from measured outputs.

The paired bootstrap resamples clips within each true class. The checkpoint,
recordings and three Gaussian replicate streams remain fixed; its interval does
not estimate variability between training runs or recording sessions.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def metrics(y, predicted, clean_predicted, n_classes):
    """Calculate accuracy, macro-F1, drop and clean-correct conditional ASR."""
    clean_correct = clean_predicted == y
    denominator = int(clean_correct.sum())
    failures = int(((predicted != y) & clean_correct).sum())
    accuracy = float((predicted == y).mean())
    return {
        "n": len(y),
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(
                y,
                predicted,
                labels=np.arange(n_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "drop_pp": 100 * (float(clean_correct.mean()) - accuracy),
        "asr": failures / denominator if denominator else np.nan,
        "asr_failures": failures,
        "clean_correct_n": denominator,
    }


def paired_ci(differences, labels, reps=10000, seed=26):
    """Stratify by class and resample paired per-clip differences, not noisy rows."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(reps, dtype=float)
    for label in np.unique(labels):
        values = np.asarray(differences)[labels == label]
        indices = rng.integers(len(values), size=(reps, len(values)))
        draws += values[indices].sum(axis=1)
    draws /= len(differences)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def validate(df, n_classes, snrs, replicates, master_seed=26, tolerance=0.1):
    """Reject missing, duplicate, unpaired or incorrectly calibrated predictions."""
    required = {
        "checkpoint_id",
        "sample_id",
        "true_label",
        "condition",
        "target_snr",
        "replicate_id",
        "master_seed",
        "pred",
        "actual_snr",
        "clip_fraction",
        "scale",
    }
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.empty or df.checkpoint_id.isna().any() or df.checkpoint_id.nunique() != 1:
        raise ValueError("Analyze exactly one checkpoint")
    if df.sample_id.isna().any() or not (df.master_seed == master_seed).all():
        raise ValueError("Missing sample IDs or a different master seed")
    if not set(df.condition).issubset({"clean", "gaussian", "fgsm"}):
        raise ValueError("Unknown evaluation condition")
    for column in ["true_label", "pred", "replicate_id"]:
        values = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError(f"{column} must contain integers")
        df[column] = values.astype(int)
    for column in ["true_label", "pred"]:
        if not df[column].between(0, n_classes - 1).all():
            raise ValueError(f"{column} outside the declared label range")
    if not np.isfinite(df[["clip_fraction", "scale"]]).all().all():
        raise ValueError("Nonfinite perturbation audit values")
    if not df.clip_fraction.between(0, 1).all() or (df.scale < 0).any():
        raise ValueError("Invalid clipping fraction or scale")

    # One clean prediction anchors every noisy comparison for that sample.
    clean = df[df.condition == "clean"].copy().sort_values("sample_id")
    if (
        clean.empty
        or clean.sample_id.duplicated().any()
        or (clean.replicate_id != -1).any()
    ):
        raise ValueError("Expected one clean row per sample; replicate_id=-1")
    if not clean.target_snr.isna().all() or set(clean.true_label) != set(
        range(n_classes)
    ):
        raise ValueError("Invalid clean rows or a missing class")
    noisy = df[df.condition != "clean"]
    expected = {("fgsm", float(s), -1) for s in snrs} | {
        ("gaussian", float(s), int(r)) for s in snrs for r in replicates
    }
    observed = {
        (c, float(s), int(r))
        for c, s, r in noisy[["condition", "target_snr", "replicate_id"]].itertuples(
            index=False, name=None
        )
    }
    if observed != expected:
        raise ValueError(
            f"Condition mismatch: missing={expected-observed}, extra={observed-expected}"
        )
    if (
        not np.isfinite(noisy.actual_snr).all()
        or ((noisy.actual_snr - noisy.target_snr).abs() > tolerance + 1e-7).any()
    ):
        raise ValueError("Measured SNR is nonfinite or outside tolerance")
    for key, group in noisy.groupby(["condition", "target_snr", "replicate_id"]):
        group = group.sort_values("sample_id")
        if (
            group.sample_id.duplicated().any()
            or group.sample_id.tolist() != clean.sample_id.tolist()
        ):
            raise ValueError(f"Unpaired sample IDs: {key}")
        if group.true_label.tolist() != clean.true_label.tolist():
            raise ValueError(f"Changed ground-truth labels: {key}")
    return clean


def analyze(
    df, n_classes, snrs=(30, 20, 10), replicates=(0, 1, 2), reps=10000, master_seed=26
):
    """Return per-replicate metrics, aggregate metrics, effects and aligned rows."""
    df = df.copy()
    clean = validate(df, n_classes, snrs, replicates, master_seed)
    y = clean.true_label.to_numpy()
    clean_predictions = clean.pred.to_numpy()
    per_replicate, groups = [], {}
    for (condition, snr, replicate), group in df.groupby(
        ["condition", "target_snr", "replicate_id"], dropna=False
    ):
        group = group.sort_values("sample_id")
        predicted = group.pred.to_numpy()
        row = metrics(y, predicted, clean_predictions, n_classes)
        if condition == "clean":
            row.update(asr=np.nan, asr_failures=np.nan)
        row.update(
            condition=condition,
            target_snr=snr,
            replicate_id=int(replicate),
            master_seed=master_seed,
            actual_snr_mean=group.actual_snr.mean(),
            actual_snr_min=group.actual_snr.min(),
            actual_snr_max=group.actual_snr.max(),
            clip_fraction_mean=group.clip_fraction.mean(),
            clip_fraction_max=group.clip_fraction.max(),
        )
        per_replicate.append(row)
        if condition != "clean":
            groups[(condition, float(snr), int(replicate))] = predicted
    per_replicate = pd.DataFrame(per_replicate)

    # Average metrics across noise streams; do not pool them as independent clips.
    summary = []
    for (condition, snr), group in per_replicate.groupby(
        ["condition", "target_snr"], dropna=False
    ):
        row = {
            "condition": condition,
            "target_snr": snr,
            "replicates": len(group),
            "n": len(y),
            "master_seed": master_seed,
        }
        for metric in ["accuracy", "macro_f1", "drop_pp", "asr"]:
            row[metric] = group[metric].mean()
            row[metric + "_std"] = (
                group[metric].std(ddof=1) if len(group) > 1 else np.nan
            )
        for metric in ["actual_snr_mean", "clip_fraction_mean"]:
            row[metric] = group[metric].mean()
        row.update(
            actual_snr_min=group.actual_snr_min.min(),
            actual_snr_max=group.actual_snr_max.max(),
            clip_fraction_max=group.clip_fraction_max.max(),
        )
        summary.append(row)

    # Positive D means FGSM has lower accuracy than Gaussian at matched SNR.
    effects = []
    for snr in snrs:
        gaussian_correct = np.stack(
            [groups[("gaussian", float(snr), r)] == y for r in replicates]
        )
        fgsm_correct = (groups[("fgsm", float(snr), -1)] == y).astype(float)
        differences = gaussian_correct.mean(axis=0) - fgsm_correct
        low, high = paired_ci(differences, y, reps, seed=master_seed)
        effects.append(
            {
                "target_snr": snr,
                "difference_pp": 100 * differences.mean(),
                "ci_low_pp": 100 * low,
                "ci_high_pp": 100 * high,
                "n": len(y),
                "gaussian_replicates": len(replicates),
                "master_seed": master_seed,
                "primary": snr == 20,
                "interpretation": (
                    "supports_direction"
                    if low > 0
                    else "opposite_direction" if high < 0 else "inconclusive"
                ),
            }
        )
    return per_replicate, pd.DataFrame(summary), pd.DataFrame(effects), clean, groups


def write_report_tables(summary, effects, out, partition):
    """Write Markdown and LaTeX tables directly from measured metrics."""
    out = Path(out)
    rows, latex = [], []
    order = {"clean": 0, "gaussian": 1, "fgsm": 2}
    records = sorted(
        summary.to_dict("records"),
        key=lambda r: (
            order[r["condition"]],
            -r["target_snr"] if pd.notna(r["target_snr"]) else 0,
        ),
    )
    for row in records:
        name = {"clean": "Clean", "gaussian": "Gaussian", "fgsm": "FGSM-sign"}[
            row["condition"]
        ]
        snr = "--" if pd.isna(row["target_snr"]) else f"{row['target_snr']:.0f}"
        accuracy = f"{100*row['accuracy']:.2f}"
        if pd.notna(row["accuracy_std"]):
            accuracy += f" ± {100*row['accuracy_std']:.2f}"
        asr = "--" if pd.isna(row["asr"]) else f"{100*row['asr']:.2f}"
        values = [
            name,
            snr,
            accuracy,
            f"{row['macro_f1']:.4f}",
            f"{row['drop_pp']:.2f}",
            asr,
        ]
        rows.append("| " + " | ".join(values) + " |")
        latex.append(" & ".join(v.replace("±", r"$\pm$") for v in values) + r" \\")
    note = (
        "Exploratory validation diagnostics; validation was also used for model selection."
        if partition == "val"
        else "Main test experiment; the primary comparison is at 20 dB."
    )
    markdown = f"# {partition.upper()} results\n\n{note}\n\n"
    markdown += "| Condition | SNR (dB) | Accuracy (%) | Macro-F1 | Drop (pp) | ASR (%) |\n|---|---:|---:|---:|---:|---:|\n"
    markdown += (
        "\n".join(rows)
        + "\n\nGaussian ± denotes sample standard deviation across three streams derived from master seed 26, not a confidence interval. ASR is conditional on clean-correct clips.\n\n"
    )
    markdown += "Paired effect: Gaussian accuracy minus FGSM accuracy. See paired_effects.csv for stratified paired bootstrap intervals.\n"
    (out / "results_table.md").write_text(markdown)
    tex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{"
            + note
            + r" Gaussian accuracy reports mean and sample standard deviation across noise streams.}",
            r"\label{tab:" + partition + r"_results}",
            r"\begin{tabular}{lrrrrr}",
            r"\hline",
            r"Condition & SNR (dB) & Accuracy (\%) & Macro-F1 & Drop (pp) & ASR (\%) \\",
            r"\hline",
            *latex,
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (out / "results_table.tex").write_text(tex)


def main():
    """Recalculate tables from a completed prediction CSV without inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--partition", choices=["test", "val"], required=True)
    args = parser.parse_args()
    meta = json.loads(args.input.with_name("evaluation_metadata.json").read_text())
    if meta["partition"] != args.partition:
        raise ValueError("Requested partition differs from prediction metadata")
    args.out.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(args.input, dtype={"sample_id": str, "checkpoint_id": str})
    per, summary, effects, _, _ = analyze(
        df,
        meta["num_classes"],
        meta["snrs"],
        meta["gaussian_replicates"],
        master_seed=meta["master_seed"],
    )
    if args.partition == "val":
        effects = effects.rename(columns={"primary": "reference_to_main_test_snr"})
    for name, table in [
        ("metrics_per_replicate", per),
        ("metrics_summary", summary),
        ("paired_effects", effects),
    ]:
        table.to_csv(args.out / (name + ".csv"), index=False)
    write_report_tables(summary, effects, args.out, args.partition)


if __name__ == "__main__":
    main()
