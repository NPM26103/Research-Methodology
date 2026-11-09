"""RMS-based candidate segmentation; inspect QC before splitting/training.

This is a project segmentation method, not the author's FFT isolator.
No model predictions are used to select segments.
"""

import argparse, html, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from .common import audio, sha


def main():
    """Find RMS-energy peaks, save candidate crops, and create a human QC gallery."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/raw/MBPWavs")
    p.add_argument("--out", default="data/processed")
    p.add_argument(
        "--settings", help="JSON mapping key to prominence/min_gap_s/peaks_s overrides"
    )
    a = p.parse_args()
    out = Path(a.out)
    if (out / "manifest.csv").exists():
        raise ValueError(
            "Use a NEW output directory when resegmenting; retain the previous version"
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "qc").mkdir(exist_ok=True)
    settings = json.loads(Path(a.settings).read_text()) if a.settings else {}
    rows = []
    summary = []
    index = [
        '<meta charset="utf-8"><title>Segmentation QC</title><h1>Segmentation QC</h1><p>Review waveforms and audio before splitting. Counts alone do not prove valid segmentation.</p>'
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(a.source)
    source_paths = {
        f.stem.lower(): f for f in source.iterdir() if f.suffix.lower() == ".wav"
    }
    for key in "0123456789abcdefghijklmnopqrstuvwxyz":
        path = source_paths.get(key)
        if path is None:
            raise ValueError("Missing WAV for key " + key)
        x, original_sr = audio(path)
        sr = 44100
        hop = 128
        window = 256
        # Build a short-time RMS envelope without using any model predictions.
        xp = np.pad(x, (window // 2, window // 2))
        frames = np.lib.stride_tricks.sliding_window_view(xp, window)[::hop]
        energy = np.sqrt(np.mean(frames**2, axis=1))
        smooth = np.convolve(energy, np.ones(3) / 3, mode="same")
        scaled = smooth / max(float(smooth.max()), 1e-12)
        cfg = settings.get(key, {})
        prominence = float(cfg.get("prominence", 0.06))
        gap = float(cfg.get("min_gap_s", 0.35))
        # Optional manual peak corrections must be decided before the split.
        if "peaks_s" in cfg:
            peaks = np.round(np.asarray(cfg["peaks_s"], dtype=float) * sr).astype(int)
        else:
            pk, _ = find_peaks(
                scaled, prominence=prominence, distance=max(1, round(gap * sr / hop))
            )
            peaks = pk * hop
        peaks = np.sort(peaks)
        count = 0
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(np.arange(len(x))[::50] / sr, x[::50], lw=0.4)
        ax.set(
            title=f"{key}: {len(peaks)} candidates (expected about 25)",
            xlabel="Time (s)",
            ylabel="Amplitude",
        )
        page = [
            '<meta charset="utf-8">',
            f"<h1>Key {html.escape(key)}</h1>",
            f'<img width="100%" src="{key}.png">',
            "<p>Record rejected candidates and listening-based reasons in the QC review before freezing the split.</p>",
        ]
        for i, peak in enumerate(peaks):
            # Preserve a fixed onset margin and a fixed 14,400-sample crop.
            start = int(peak) - 2400
            end = start + 14400
            if start < 0 or end > len(x):
                continue
            sid = f"phone_{key}_{i:03d}"
            clip = x[start:end].astype(np.float32)
            if np.sqrt(np.mean(clip**2)) < 1e-8:
                continue
            rel = Path("clips") / key / f"{sid}.wav"
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(dest, sr, clip)
            rows.append(
                dict(
                    sample_id=sid,
                    recording_id=key,
                    key=key,
                    path=rel.as_posix(),
                    start_sample=start,
                    end_sample=end,
                    peak_sample=int(peak),
                    start_s=start / sr,
                    sample_rate=sr,
                    num_samples=len(clip),
                    sha256=sha(dest),
                    keep=1,
                )
            )
            page.append(
                f'<p>{sid} at {peak/sr:.3f}s</p><audio controls preload="none" src="../{rel.as_posix()}"></audio>'
            )
            ax.axvline(peak / sr, color="#b54a36", lw=0.5)
            count += 1
        fig.tight_layout()
        fig.savefig(out / "qc" / f"{key}.png", dpi=120)
        plt.close(fig)
        (out / "qc" / f"{key}.html").write_text("\n".join(page), encoding="utf8")
        record = dict(
            key=key,
            original_sr=original_sr,
            raw_sha256=sha(path),
            candidates=len(peaks),
            saved_clips=count,
            expected=25,
            prominence=prominence,
            min_gap_s=gap,
            manual_peaks=bool("peaks_s" in cfg),
        )
        summary.append(record)
        index.append(f'<p><a href="{key}.html">Key {key}: {count} clips</a></p>')
        print(f"{key}: {count} clips", flush=True)
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "qc" / "segmentation_summary.csv", index=False)
    (out / "qc" / "index.html").write_text("\n".join(index), encoding="utf8")
    (out / "segmentation_config.json").write_text(
        json.dumps(
            {
                "method": "RMS prominence candidates",
                "sample_rate": 44100,
                "window": 256,
                "hop": 128,
                "smooth_frames": 3,
                "before": 2400,
                "clip_samples": 14400,
                "overrides": settings,
                "raw_source": str(source),
            },
            indent=2,
        )
    )
    print(
        "Inspect qc/index.html, listen to the required candidates, and record the QC review before freezing."
    )


if __name__ == "__main__":
    main()
