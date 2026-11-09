import hashlib, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


def sha(path):
    """Hash a file in chunks so raw recordings do not need to fit in memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audio(path, sample_rate=44100):
    """Load PCM/float WAV, convert to mono float32, and resample if required."""
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.signedinteger):
        x = x.astype(np.float32) / (2 ** (x.dtype.itemsize * 8 - 1))
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128) / 128
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    if not np.isfinite(x).all() or np.abs(x).max() > 1:
        raise ValueError(f"Invalid amplitude: {path}")
    if sr != sample_rate:
        g = math.gcd(sr, sample_rate)
        x = resample_poly(x, sample_rate // g, sr // g).astype(np.float32)
    if np.abs(x).max() > 1:
        raise ValueError(f"Resampling exceeds range: {path}")
    return x, int(sr)


def seed_all(seed):
    """Initialize Python, NumPy and Torch; request deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device(name):
    """Resolve the requested device and fail early if CUDA is unavailable."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in ["cpu", "cuda"]:
        raise ValueError("Use cpu or cuda; MPS frontend is not validated")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose CPU or a GPU runtime")
    return torch.device(name)


def load_partition(processed, part, expected=None):
    """Load only the requested split, checking metadata and every clip hash."""
    root = Path(processed).resolve()
    hashes = {
        "manifest_sha256": sha(root / "manifest.csv"),
        "split_sha256": sha(root / "split.csv"),
        "label_map_sha256": sha(root / "label_map.json"),
    }
    if expected:
        for key, value in hashes.items():
            if expected[key] != value:
                raise ValueError("Dataset metadata changed: " + key)
    manifest = pd.read_csv(root / "manifest.csv", dtype={"sample_id": str, "key": str})
    split = pd.read_csv(root / "split.csv", dtype={"sample_id": str})
    mapping = json.loads((root / "label_map.json").read_text())
    chosen = (
        split[split["split"] == part]
        .merge(manifest, on="sample_id", validate="one_to_one")
        .sort_values("sample_id")
    )
    if chosen.empty:
        raise ValueError("Empty partition " + part)
    xs = []
    ys = []
    for r in chosen.to_dict("records"):
        path = (root / r["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Clip path escapes data root")
        if sha(path) != r["sha256"]:
            raise ValueError("Clip changed: " + str(path))
        sr, x = wavfile.read(path)
        if sr != 44100 or x.ndim != 1 or len(x) != 14400:
            raise ValueError("Clip shape/sample-rate mismatch")
        if not np.isfinite(x).all() or np.abs(x).max() > 1:
            raise ValueError("Invalid clip")
        xs.append(x.astype(np.float32))
        ys.append(mapping[r["key"]])
    return (
        torch.from_numpy(np.stack(xs)),
        torch.tensor(ys, dtype=torch.long),
        chosen.sample_id.tolist(),
        mapping,
        hashes,
    )
