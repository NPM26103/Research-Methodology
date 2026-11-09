"""Data preparation, training, evaluation and report exports."""

import argparse, json
from pathlib import Path
from .io import ROOT, own_run, call, freeze_checkpoint
from .safety import save_json
from .data import prepare, freeze, verify_frozen


def local_data_root(data, run):
    """Keep this run cache separate from persistent outputs and unrelated data."""
    data = Path(data).resolve()
    run = Path(run).resolve()
    if data == run or data.is_relative_to(run) or run.is_relative_to(data):
        raise ValueError("Use separate data-root and run folders")
    marker = data / "rsm_data_owner.json"
    if not marker.exists():
        if data.exists() and any(data.iterdir()):
            raise ValueError("Choose a NEW empty data cache folder")
        data.mkdir(parents=True, exist_ok=True)
        save_json(marker, {"run": str(run)})
    if json.loads(marker.read_text()) != {"run": str(run)}:
        raise ValueError("Data cache belongs to another run")
    return data


def main():
    """Dispatch one explicit workflow stage; never select a model by test performance."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "prepare",
            "freeze",
            "train",
            "sanity",
            "main",
            "diagnostics",
            "export-main",
            "export-validation",
            "export-all",
        ],
    )
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--workspace-zip", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "configs/run.json")
    p.add_argument("--qc-review", type=Path)
    p.add_argument("--allow-data-revision", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--session-epochs", type=int)
    p.add_argument("--include-checkpoint", action="store_true")
    a = p.parse_args()
    run = own_run(a.run)
    data = local_data_root(a.data_root, run)
    if a.stage == "prepare":
        prepare(run, data, a.workspace_zip)
    elif a.stage == "freeze":
        if a.qc_review is None and not (run / "frozen_protocol.json").exists():
            raise ValueError("Provide --qc-review after actual listening")
        freeze(run, data, a.config, a.qc_review, a.allow_data_revision)
    else:
        protocol, processed = verify_frozen(run, data)
        if a.stage == "train":
            if (run / "checkpoint_lock.json").exists():
                freeze_checkpoint(run)
                print("Checkpoint locked; keeping completed training.")
                return
            args = [
                "--processed",
                processed,
                "--config",
                run / "training_config.json",
                "--run",
                run / "training",
                "--device",
                a.device,
            ]
            if (run / "training/last.pt").exists():
                args += ["--resume"]
            if a.session_epochs is not None:
                args += ["--session-epochs", a.session_epochs]
            call(run, "rsm.train", *args)
        elif a.stage.startswith("export-"):
            from .export import export_results

            export_results(run, a.stage.removeprefix("export-"), a.include_checkpoint)
        else:
            from .evaluation import preflight, evaluate

            if a.stage == "sanity":
                preflight(run, processed, a.device)
            else:
                evaluate(
                    run,
                    processed,
                    a.device,
                    "test" if a.stage == "main" else "val",
                    protocol["study"],
                )


if __name__ == "__main__":
    main()
