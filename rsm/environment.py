"""Install pinned libraries while preserving a compatible Colab torch trio."""

import argparse, importlib.metadata, json, subprocess, sys, tempfile
from pathlib import Path


def main():
    """Install pinned scientific packages while preserving a compatible Torch GPU trio."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.version_info < (3, 11):
        raise RuntimeError("Use Python 3.11 or newer")
    trio = ["torch", "torchvision", "torchaudio"]
    try:
        versions = {k: importlib.metadata.version(k) for k in trio}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Install compatible torch, torchvision and torchaudio builds first. "
            "Colab GPU runtimes normally include them."
        ) from exc
    prior = a.run / "training/environment.json"
    if prior.exists():
        saved = json.loads(prior.read_text())
        expected = {
            "torch": saved["torch"],
            **{k: saved["packages"][k] for k in trio[1:]},
        }
        if versions != expected:
            raise RuntimeError(
                "Restore the training GPU environment before resume: "
                + json.dumps(expected)
            )
    with tempfile.TemporaryDirectory(prefix="rsm_deps_") as d:
        constraint = Path(d) / "torch_constraints.txt"
        constraint.write_text("".join(f"{k}=={v}\n" for k, v in versions.items()))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(root / "requirements.txt"),
                "-c",
                str(constraint),
            ],
            check=True,
        )
    check = """import json,torch,torchaudio,torchvision,timm
from rsm.model import Recognizer
from rsm.common import device
device(sys.argv[1])
torch.set_num_threads(2)
m=Recognizer(json.load(open('configs/run.json')),36).eval()
with torch.no_grad():z=m(torch.zeros(1,14400))
assert z.shape==(1,36) and torch.isfinite(z).all()
print('Imports and CoAtNet forward: PASS; CUDA available:',torch.cuda.is_available())
print(torch.__version__,torchaudio.__version__,torchvision.__version__,timm.__version__)
"""
    subprocess.run(
        [sys.executable, "-B", "-c", "import sys\n" + check, a.device],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
