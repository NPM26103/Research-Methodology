"""Read WAV metadata/scale without changing files. Requires scipy and numpy."""

import argparse, csv, hashlib
from pathlib import Path
import numpy as np
from scipy.io import wavfile


def main():
    """Audit recording sample rates, durations, amplitudes and hashes before segmentation."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    paths = sorted(Path(a.source).rglob("*"))
    paths = [x for x in paths if x.is_file() and x.suffix.lower() == ".wav"]
    if not paths:
        raise ValueError("No WAV files found")
    rows = []
    for path in paths:
        sr, x = wavfile.read(path)
        channels = 1 if x.ndim == 1 else x.shape[1]
        dtype = str(x.dtype)
        if np.issubdtype(x.dtype, np.signedinteger):
            z = x.astype(np.float64) / (2 ** (x.dtype.itemsize * 8 - 1))
        elif x.dtype == np.uint8:
            z = (x.astype(np.float64) - 128) / 128
        else:
            z = x.astype(np.float64)
        if len(z) == 0 or not np.isfinite(z).all():
            raise ValueError(f"Empty/nonfinite WAV: {path}")
        rows.append(
            dict(
                recording_id=path.stem,
                key=path.stem.lower(),
                path=str(path),
                sample_rate=sr,
                channels=channels,
                dtype=dtype,
                num_frames=len(x),
                duration_s=len(x) / sr,
                rms=float(np.sqrt(np.mean(z * z))),
                peak=float(np.abs(z).max()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw_manifest.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    expected = set("0123456789abcdefghijklmnopqrstuvwxyz")
    got = {r["key"] for r in rows}
    print(
        "WAV count:",
        len(rows),
        "missing labels:",
        sorted(expected - got),
        "extra labels:",
        sorted(got - expected),
    )
    print("Sample rates:", sorted({r["sample_rate"] for r in rows}))
    print("Channels:", sorted({r["channels"] for r in rows}))
    print("Inspect/listen before segmentation. Metadata is not manual QC approval.")


if __name__ == "__main__":
    main()
