"""Owned run directories, immutable code/config and preserved results."""

import json, subprocess, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
from .safety import sha, save_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "rsm-4.0.0"


def now():
    """Use UTC timestamps for portable run and export records."""
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    """Fingerprint executable code, configs and dependency specifications."""
    paths = [
        p
        for folder in ["rsm", "configs"]
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in [".py", ".json"]
    ] + [ROOT / "requirements.txt"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def own_run(run):
    """Create an owned run or verify that its code/config has not changed."""
    run = Path(run).resolve()
    marker = run / "run_identity.json"
    if not marker.exists():
        if run.exists() and any(run.iterdir()):
            raise ValueError(
                "Run folder contains unrelated files; choose a NEW RUN_NAME"
            )
        run.mkdir(parents=True, exist_ok=True)
        save_json(
            marker,
            {
                "project_version": VERSION,
                "created_utc": now(),
                "source_sha256": source_hashes(),
            },
        )
    identity = json.loads(marker.read_text())
    if (
        identity["project_version"] != VERSION
        or identity["source_sha256"] != source_hashes()
    ):
        raise ValueError(
            "Source/config changed; use the original package to resume or a NEW run"
        )
    return run


def call(run, module, *args):
    """Stream a subprocess to the notebook and append its output to the run log."""
    cmd = [sys.executable, "-u", "-B", "-m", module, *map(str, args)]
    with (run / "run.log").open("a", encoding="utf-8") as log:
        line = "\n" + now() + " " + json.dumps(cmd) + "\n"
        log.write(line)
        print(line, flush=True)
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, cmd)


def archive_folder(folder, dest):
    """Write a recoverable ZIP through a temporary file before renaming it."""
    folder, dest = Path(folder), Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(folder))
    temp.replace(dest)
    return sha(dest)


def source_snapshot(dest):
    """Archive source files without including data, runs or virtual environments."""
    paths = [
        ROOT / name
        for name in ["RsM.ipynb", "README.md", "requirements.txt", ".gitignore"]
    ]
    for folder in ["rsm", "configs", "docs", "tests"]:
        paths.extend((ROOT / folder).rglob("*"))
    with zipfile.ZipFile(dest, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):
            if (
                p.is_file()
                and "__pycache__" not in p.parts
                and (
                    p.suffix in [".py", ".json", ".md", ".txt", ".ipynb"]
                    or p.name == ".gitignore"
                )
            ):
                z.write(p, p.relative_to(ROOT))


def new_attempt(run, name):
    """Keep each evaluation attempt in a new timestamped folder."""
    path = (
        run
        / "results"
        / name
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def latest_completed(run, name):
    """Find the latest complete attempt; ignore interrupted attempts."""
    for path in sorted((run / "results" / name).glob("*"), reverse=True):
        if path.is_dir() and (path / "completion.json").exists():
            return path
    return None


def freeze_checkpoint(run):
    """Lock the selected checkpoint only after training has completed."""
    if not (run / "training/training_complete.json").exists():
        raise ValueError("Training incomplete/paused; resume before evaluation")
    ck = run / "training/best.pt"
    value = {"checkpoint_sha256": sha(ck)}
    lock = run / "checkpoint_lock.json"
    if lock.exists():
        if json.loads(lock.read_text()) != value:
            raise ValueError("Frozen best checkpoint changed")
    else:
        save_json(lock, value)
    return ck, value["checkpoint_sha256"]
