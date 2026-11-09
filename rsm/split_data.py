import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    """Create one stratified within-recording split after QC; refuse to overwrite it."""
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--seed", type=int, default=26)
    a = p.parse_args()
    root = Path(a.processed)
    if (root / "split.csv").exists():
        raise ValueError("Split already exists; retain it instead of regenerating")
    df = pd.read_csv(root / "manifest.csv", dtype={"sample_id": str, "key": str})
    if not df.keep.isin([0, 1]).all():
        raise ValueError("keep must be 0 or 1")
    df = df[df.keep == 1].copy()
    classes = list("0123456789abcdefghijklmnopqrstuvwxyz")
    if set(df.key) != set(classes):
        raise ValueError("All 36 classes must remain represented")
    if df.sample_id.duplicated().any() or df.sha256.duplicated().any():
        raise ValueError("Duplicate samples/content found")
    for key, g in df.groupby("recording_id"):
        g = g.sort_values("start_sample")
        if (g.start_sample.to_numpy()[1:] < g.end_sample.to_numpy()[:-1]).any():
            raise ValueError("Overlapping crops: inspect " + key)
    rng = np.random.default_rng(a.seed)
    rows = []
    for key in classes:
        ids = df[df.key == key].sample_id.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        if n < 5:
            raise ValueError("Too few valid clips for " + key)
        nv = max(1, round(n * 0.2))
        nt = max(1, round(n * 0.2))
        for i, sid in enumerate(ids):
            rows.append(
                {
                    "sample_id": sid,
                    "split": "test" if i < nt else "val" if i < nt + nv else "train",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(root / "split.csv", index=False)
    (root / "label_map.json").write_text(
        json.dumps({k: i for i, k in enumerate(classes)}, indent=2)
    )
    counts = (
        result.merge(df[["sample_id", "key"]], on="sample_id")
        .groupby(["key", "split"])
        .size()
        .unstack(fill_value=0)
    )
    counts.to_csv(root / "split_counts.csv")
    print(counts)
    print("Totals:", result["split"].value_counts().to_dict())
    (root / "split_config.json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "method": "within-class shuffle, round(20%) val/test, remainder train",
                "scope": "within-recording only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
