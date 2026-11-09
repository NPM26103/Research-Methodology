"""Checks for the repository layout and notebook entry point."""

import ast
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from rsm.io import ROOT, source_snapshot
from rsm.notebook import Experiment


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class RepositoryTests(unittest.TestCase):
    def test_source_snapshot_excludes_generated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = [
                "RsM.ipynb",
                "README.md",
                "requirements.txt",
                ".gitignore",
                "rsm/model.py",
                "configs/run.json",
                "docs/METHODS.md",
            ]
            excluded = [
                "data/manifest.json",
                "runs/private/config.json",
                ".venv/private.py",
                "unrelated/notes.md",
                "rsm/__pycache__/cached.py",
            ]
            for name in kept + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            archive = root / "source.zip"
            with patch("rsm.io.ROOT", root):
                source_snapshot(archive)
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), set(kept))
                self.assertIsNone(bundle.testzip())

    def test_qc_requires_an_explicit_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = Experiment(root / "run", root / "data", device="cpu")
            with patch.object(experiment, "run") as run:
                for reviewer, notes, reviewed in [
                    ("", "notes", True),
                    ("reviewer", "", True),
                    ("reviewer", "notes", False),
                ]:
                    with self.assertRaises(ValueError):
                        experiment.freeze(reviewer, notes, reviewed)
                run.assert_not_called()
                self.assertFalse((root / "run/manual_qc_input.json").exists())

    def test_notebook_runs_from_a_local_checkout(self):
        notebook = json.loads((ROOT / "RsM.ipynb").read_text())
        bootstrap = next(c for c in notebook["cells"] if c["cell_type"] == "code")
        namespace = {}
        original_path = sys.path[:]
        try:
            with working_directory(ROOT), patch("subprocess.run") as command:
                exec(
                    compile("".join(bootstrap["source"]), "RsM.ipynb", "exec"),
                    namespace,
                )
                command.assert_not_called()
                self.assertEqual(namespace["ROOT"], ROOT)
        finally:
            sys.path[:] = original_path

    def test_notebook_is_clean_and_exports_main_first(self):
        notebook = json.loads((ROOT / "RsM.ipynb").read_text())
        stages = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "experiment"
                    and node.func.attr in {"run", "export"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    stages.append((node.func.attr, node.args[0].value))
        self.assertLess(
            stages.index(("export", "main")), stages.index(("run", "diagnostics"))
        )

    def test_failed_export_does_not_return_a_stale_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = Experiment(root / "run", root / "data", device="cpu")
            exports = root / "run/exports"
            exports.mkdir(parents=True)
            (exports / "old_ALL_report_bundle_000.zip").write_bytes(b"old")
            error = subprocess.CalledProcessError(1, ["rsm.pipeline"])
            with patch.object(experiment, "run", side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    experiment.export("all")


if __name__ == "__main__":
    unittest.main()
