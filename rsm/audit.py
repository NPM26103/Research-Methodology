"""Read-only audit of the frozen dataset and previously saved predictions."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from .safety import sha, save_json


def audit_data(root, ck, out):
    """Audit file integrity, labels, splits, crop overlap and duplicate audio content."""
    root, out = Path(root).resolve(), Path(out)
    hashes = {
        k + "_sha256": sha(root / f)
        for k, f in [
            ("manifest", "manifest.csv"),
            ("split", "split.csv"),
            ("label_map", "label_map.json"),
        ]
    }
    if hashes != ck["data_hashes"]:
        raise ValueError("Processed data hashes do not match the checkpoint")
    m = pd.read_csv(
        root / "manifest.csv", dtype={"sample_id": str, "key": str, "recording_id": str}
    )
    s = pd.read_csv(root / "split.csv", dtype={"sample_id": str})
    mapping = json.loads((root / "label_map.json").read_text())
    if mapping != ck["label_map"] or set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Invalid or changed label map")
    if m.sample_id.duplicated().any() or s.sample_id.duplicated().any():
        raise ValueError("Duplicate sample ID")
    if not m.keep.isin([0, 1]).all() or set(s.split) != {"train", "val", "test"}:
        raise ValueError("Invalid keep/split values")
    kept = m[m.keep == 1]
    if set(kept.sample_id) != set(s.sample_id) or not set(kept.key).issubset(mapping):
        raise ValueError("Manifest and split mismatch")
    for part in ["train", "val"]:
        actual = set(s.loc[s.split == part, "sample_id"])
        saved = ck[part + "_ids"]
        if actual != set(saved) or len(saved) != len(set(saved)):
            raise ValueError(f"{part} sample IDs differ from checkpoint")
    for _, g in kept.groupby("recording_id"):
        g = g.sort_values("start_sample")
        if (g.start_sample.to_numpy()[1:] < g.end_sample.to_numpy()[:-1]).any():
            raise ValueError("Overlapping clip intervals")
    raw_hashes, rows = [], []
    for r in m.to_dict("records"):
        p = (root / r["path"]).resolve()
        if not p.is_relative_to(root) or sha(p) != r["sha256"]:
            raise ValueError("Clip path/hash mismatch: " + r["sample_id"])
        sr, x = wavfile.read(p)
        if sr != 44100 or x.shape != (14400,) or x.dtype != np.float32:
            raise ValueError("Unexpected waveform format: " + r["sample_id"])
        if (
            not np.isfinite(x).all()
            or np.abs(x).max() > 1
            or np.mean(x.astype(np.float64) ** 2) <= 1e-16
        ):
            raise ValueError("Invalid/silent clip: " + r["sample_id"])
        if int(r["end_sample"]) - int(r["start_sample"]) != len(x):
            raise ValueError("Crop interval length mismatch")
        if r["keep"] == 1:
            import hashlib

            raw_hashes.append(hashlib.sha256(x.tobytes()).hexdigest())
        rows.append(
            {
                "sample_id": r["sample_id"],
                "rms": float(np.sqrt(np.mean(x * x))),
                "peak": float(np.abs(x).max()),
            }
        )
    if len(raw_hashes) != len(set(raw_hashes)):
        raise ValueError("Identical waveform content among accepted clips")
    joined = s.merge(kept[["sample_id", "key"]], on="sample_id", validate="one_to_one")
    counts = joined.groupby(["key", "split"]).size().unstack(fill_value=0)
    if len(counts) != len(mapping) or (counts <= 0).any().any():
        raise ValueError("A class is missing from a partition")
    counts.to_csv(out / "class_split_counts.csv")
    pd.DataFrame(rows).to_csv(out / "clip_measurements.csv", index=False)
    cfg = root / "split_config.json"
    result = {
        "passed": True,
        "clips": len(m),
        "accepted": len(kept),
        "classes": len(mapping),
        "split_counts": s.split.value_counts().to_dict(),
        "data_hashes": hashes,
        "split_config": json.loads(cfg.read_text()) if cfg.exists() else None,
        "training_seed": ck["config"].get("seed"),
        "test_waveforms": "read for integrity checks only; no test model inference",
        "scope": "Within-recording splits; automatic audit does not certify manual listening/QC or cross-session independence.",
    }
    save_json(out / "data_audit.json", result)
    return m, s, mapping, hashes


def compare_table(saved, new, keys, cols, out):
    old = pd.read_csv(saved)
    required = set(keys) | set(cols)
    if not required.issubset(old.columns):
        raise ValueError(
            f"Missing metrics columns in {saved}: {required-set(old.columns)}"
        )
    a, b = old.copy(), new.copy()
    # An identical sentinel is used only for clean's missing target SNR.
    for k in keys:
        a[k] = a[k].fillna("__CLEAN__")
        b[k] = b[k].fillna("__CLEAN__")
    merged = a.merge(
        b,
        on=keys,
        suffixes=("_saved", "_recomputed"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    rows = []
    for r in merged.to_dict("records"):
        for col in cols:
            x, y = r.get(col + "_saved", np.nan), r.get(col + "_recomputed", np.nan)
            ok = r["_merge"] == "both" and bool(
                np.isclose(x, y, atol=1e-8, rtol=1e-7, equal_nan=True)
            )
            rows.append(
                {
                    **{k: r[k] for k in keys},
                    "metric": col,
                    "saved": x,
                    "recomputed": y,
                    "matches": ok,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    return bool(table.matches.all())
