"""Export each study phase independently; exporting never runs inference."""

import json
import zipfile
from pathlib import Path
from .io import freeze_checkpoint, now
from .safety import sha

SCOPES = {
    "main": ["validation_sanity", "main_test"],
    "validation": ["validation_sanity", "exploratory_validation"],
    "all": ["validation_sanity", "main_test", "exploratory_validation"],
}


def export_results(run: Path, scope="main", include_checkpoint=False) -> Path:
    """Export verified results; main export does not require validation diagnostics."""
    from .evaluation import reuse

    if scope not in SCOPES:
        raise ValueError("Export scope must be main, validation or all")
    checkpoint, digest = freeze_checkpoint(run)
    outputs = {name: reuse(run, name, digest) for name in SCOPES[scope]}
    missing = [name for name, path in outputs.items() if path is None]
    if missing:
        raise ValueError("Complete these stages before export: " + ", ".join(missing))

    # Verify that every included evaluation used the locked checkpoint and data.
    protocol = json.loads((run / "frozen_protocol.json").read_text())
    evaluations = []
    for name, folder in outputs.items():
        metadata_path = folder / "evaluation_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (
                metadata["checkpoint_sha256"] != digest
                or metadata["data_hashes"] != protocol["data_hashes"]
            ):
                raise ValueError("Different checkpoint or dataset in " + name)
            complete = json.loads((folder / "completion.json").read_text())
            evaluations.append(
                {
                    "name": name,
                    "partition": metadata["partition"],
                    "rows": complete["prediction_rows"],
                    "checkpoint_sha256": digest,
                    "master_seed": metadata["master_seed"],
                    "data_hashes": metadata["data_hashes"],
                }
            )

    export_directory = run / "exports"
    export_directory.mkdir(exist_ok=True)
    timestamp = now().replace(":", "").replace("-", "").replace(".", "_")
    label = {
        "main": "MAIN_results",
        "validation": "VALIDATION_diagnostics",
        "all": "ALL_report_bundle",
    }[scope]
    archive = export_directory / f"{run.name}_{label}_{timestamp}.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        # Capture the protocol, provenance and training evidence for the report.
        for name in [
            "run_identity.json",
            "frozen_protocol.json",
            "checkpoint_lock.json",
            "training_config.json",
            "data_archive.json",
            "run.log",
            "source_snapshot.zip",
        ]:
            if (run / name).exists():
                bundle.write(run / name, name)
        for name in ["data_metadata", "training"]:
            for path in sorted((run / name).rglob("*")):
                if (
                    path.is_file()
                    and path.suffix not in [".pt", ".tmp"]
                    and "__pycache__" not in path.parts
                ):
                    bundle.write(path, path.relative_to(run))
        for name, folder in outputs.items():
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    bundle.write(
                        path, Path("results") / name / path.relative_to(folder)
                    )
        for name in ["raw_manifest.csv", "raw_source.json"]:
            path = run / "artifacts" / name
            if path.exists():
                bundle.write(path, Path("data_metadata") / name)
        metadata = {
            "created_utc": now(),
            "export_scope": scope,
            "master_seed": 26,
            "checkpoint_sha256": digest,
            "includes_checkpoint": include_checkpoint,
            "contains_validation_diagnostics": scope != "main",
            "results": evaluations,
            "interpretation": "Main test and exploratory validation are separate analyses; do not pool their predictions.",
        }
        bundle.writestr("bundle_metadata.json", json.dumps(metadata, indent=2))
        bundle.writestr(
            "READ_ME_FIRST.md",
            "# RsM results: " + scope + "\n\n"
            "Master seed: 26. This export contains measured outputs, training history, figures, tables, data/QC metadata and a source snapshot.\n\n"
            "Main: test SNR 30/20/10 dB; primary comparison at 20 dB. "
            "Validation: exploratory SNR 60/50/40/35/30/20/10 dB; this partition also selected the checkpoint.\n\n"
            "Keep training/best.pt, training/last.pt and artifacts/processed_dataset.zip in the run folder. "
            "The full dataset and checkpoints are omitted by default to keep this archive manageable.\n",
        )
        if include_checkpoint:
            bundle.write(checkpoint, "training/best.pt")
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Export ZIP integrity check failed")
    print("EXPORTED:", archive, flush=True)
    print("SHA256:", sha(archive), flush=True)
    return archive
