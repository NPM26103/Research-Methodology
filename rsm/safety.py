"""Filesystem guards: no writes to checkpoints, processed data or old results."""

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def sha(path):
    """Calculate SHA-256 without loading an entire file into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    """Write a new JSON record; refuse accidental overwrites and nonfinite numbers."""
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def snapshot(roots):
    """Record size/hash evidence for protected input files."""
    result = {}
    for name, root in roots.items():
        root = Path(root)
        if not root.exists():
            result[name] = {"__MISSING__": True}
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        entries = {}
        for p in paths:
            if p.is_symlink():
                raise ValueError(
                    f"Symlink in protected inputs; supply ordinary files: {p}"
                )
            if p.is_file():
                key = "." if root.is_file() else p.relative_to(root).as_posix()
                entries[key] = {"bytes": p.stat().st_size, "sha256": sha(p)}
        result[name] = entries
    return result


def changes(before, after):
    """Return the specific files changed between two snapshots."""
    result = []
    for root in sorted(set(before) | set(after)):
        a, b = before.get(root, {}), after.get(root, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                result.append(
                    {
                        "root": root,
                        "path": key,
                        "before": a.get(key),
                        "after": b.get(key),
                    }
                )
    return result


def safe_extract(archive, dest, prefix=""):
    """Extract only selected members into a new directory, rejecting unsafe names."""
    dest = Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(f"Extraction target must be new: {dest}")
    with zipfile.ZipFile(archive) as z:
        plan, seen = [], set()
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(prefix):
                continue
            name = info.filename[len(prefix) :]
            posix = PurePosixPath(name)
            target = (dest / name).resolve()
            if (
                "\\" in name
                or posix.is_absolute()
                or ".." in posix.parts
                or not target.is_relative_to(dest)
                or target == dest
                or stat.S_ISLNK(info.external_attr >> 16)
                or target in seen
            ):
                raise ValueError(f"Unsafe or duplicate ZIP member: {info.filename}")
            seen.add(target)
            plan.append((info, target))
        if not plan:
            raise ValueError(f"No matching files in {archive}; prefix={prefix!r}")
        dest.mkdir(parents=True, exist_ok=False)
        for info, target in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    return dest
