"""Notebook interface to the command-line pipeline."""

import json
import subprocess
import sys
from pathlib import Path

from .io import ROOT, VERSION, latest_completed


class Experiment:
    def __init__(self, run_dir, data_root, device="cuda"):
        self.run_dir = Path(run_dir).resolve()
        self.data_root = Path(data_root).resolve()
        self.processed = self.data_root / "processed"
        self.device = device
        study = json.loads((ROOT / "configs/study.json").read_text())
        config = json.loads((ROOT / "configs/run.json").read_text())
        seeds = [
            study[k]
            for k in ("master_seed", "split_seed", "training_seed", "bootstrap_seed")
        ]
        if study["project_version"] != VERSION or any(
            s != 26 for s in seeds + [config["seed"]]
        ):
            raise ValueError(
                "Expected the seed-26 configuration for this source version"
            )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu or cuda")

    def _command(self, module, *args):
        command = [sys.executable, "-u", "-B", "-m", module, *map(str, args)]
        with subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)

    def setup(self):
        """Install scientific dependencies and check the requested device."""
        self._command("rsm.environment", "--run", self.run_dir, "--device", self.device)

    def run(self, stage, *options):
        """Run a stage in a fresh process so dependency imports stay consistent."""
        self._command(
            "rsm.pipeline",
            stage,
            "--run",
            self.run_dir,
            "--data-root",
            self.data_root,
            "--device",
            self.device,
            *options,
        )

    def table(self, name):
        import pandas as pd

        return pd.read_csv(self.processed / name, dtype={"sample_id": str, "key": str})

    def review_audio(self, key="all", extra_ids=()):
        """Play the fixed QC subset at its recorded amplitude."""
        from IPython.display import Audio, Image, display
        from scipy.io import wavfile
        from .data import qc_sample_ids

        manifest = self.table("manifest.csv")
        ids = set(qc_sample_ids(self.processed)) | set(extra_ids)
        if not ids.issubset(set(manifest.sample_id)):
            raise ValueError("Unknown sample ID in the review list")
        selected = manifest[manifest.sample_id.isin(ids)]
        if key != "all":
            selected = selected[selected.key == key]
            figure = self.processed / "qc" / f"{key}.png"
            if figure.is_file():
                display(Image(filename=str(figure), width=1000))
        if selected.empty:
            raise ValueError("No clips match this key")
        print(f"QC clips: {len(selected)}")
        for row in selected.sort_values("sample_id").to_dict("records"):
            print(row["sample_id"])
            rate, waveform = wavfile.read(self.processed / row["path"])
            display(Audio(waveform, rate=rate, normalize=False))

    def freeze(
        self,
        reviewer,
        notes,
        reviewed=False,
        rejected=None,
        extra_ids=(),
        allow_revision=False,
    ):
        """Record completed audio review before fixing the data split."""
        if (self.run_dir / "frozen_protocol.json").exists():
            self.run("freeze")
            return
        if not reviewed or not reviewer.strip() or not notes.strip():
            raise ValueError("Complete the audio review and enter reviewer/notes first")
        from .data import qc_sample_ids

        review = {
            "reviewer": reviewer.strip(),
            "notes": notes.strip(),
            "reviewed_sample_ids": sorted(
                set(qc_sample_ids(self.processed)) | set(extra_ids)
            ),
            "rejected": rejected or {},
        }
        path = self.run_dir / "manual_qc_input.json"
        path.write_text(json.dumps(review, indent=2), encoding="utf-8")
        options = ["--qc-review", path]
        if allow_revision:
            options.append("--allow-data-revision")
        self.run("freeze", *options)

    def show_training(self):
        from IPython.display import Image, display

        folder = self.run_dir / "training"
        figure = folder / "training_curves.png"
        if figure.exists():
            display(Image(filename=str(figure), width=1000))
        complete = folder / "training_complete.json"
        if not complete.exists():
            raise RuntimeError(
                "Training is incomplete; rerun the training cell to resume"
            )
        print(json.dumps(json.loads(complete.read_text()), indent=2))

    def result_dir(self, stage):
        folder = latest_completed(self.run_dir, stage)
        if folder is None:
            raise FileNotFoundError(f"No completed result for {stage}")
        return folder

    def show_results(self, stage, tables, figures):
        import pandas as pd
        from IPython.display import Image, display

        folder = self.result_dir(stage)
        for name in tables:
            print(name)
            display(pd.read_csv(folder / name))
        for name in figures:
            display(Image(filename=str(folder / name), width=1000))
        return folder

    def show_example(self, index=0):
        from IPython.display import Audio, Image, display
        from scipy.io import wavfile

        folder = self.result_dir("exploratory_validation") / "examples"
        figures = sorted(folder.glob("*_comparison.png"))
        if not 0 <= index < len(figures):
            raise IndexError(f"Choose an index from 0 to {len(figures) - 1}")
        figure = figures[index]
        display(Image(filename=str(figure), width=1000))
        sample_id = figure.name.removesuffix("_comparison.png")
        for path in sorted(folder.glob(sample_id + "_*.wav")):
            print(path.name)
            rate, waveform = wavfile.read(path)
            display(Audio(waveform, rate=rate, normalize=False))

    def export(self, scope="all", include_checkpoint=False):
        labels = {
            "main": "MAIN_results",
            "validation": "VALIDATION_diagnostics",
            "all": "ALL_report_bundle",
        }
        if scope not in labels:
            raise ValueError("Export scope must be main, validation or all")
        options = ["--include-checkpoint"] if include_checkpoint else []
        self.run("export-" + scope, *options)
        archive = sorted((self.run_dir / "exports").glob(f"*_{labels[scope]}_*.zip"))[
            -1
        ]
        print(archive)
        return archive
