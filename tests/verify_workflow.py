"""Integration tests for the submission pipeline, never research measurements.

The classifier is deliberately a tiny CNN trained for three epochs. The QC
record is a labelled software fixture, not a claim that a human listened.
Use --data-source public to test download/segmentation without existing data.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from rsm.io import ROOT, own_run, latest_completed, freeze_checkpoint
from rsm.data import prepare, freeze, qc_sample_ids, verify_frozen
from rsm.pipeline import local_data_root
from rsm.safety import sha, snapshot, save_json, safe_extract
from rsm.seeding import derived_seed
from rsm.metrics import paired_ci


def expect_rejection(operation):
    """A guard is useful only if the invalid operation is actually rejected."""
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Invalid operation was accepted")


def make_synthetic_data(processed):
    """Build 36 distinct classes with five small waveforms per class."""
    (processed / "clips").mkdir(parents=True)
    rng = np.random.default_rng(26)
    rows = []
    for label, key in enumerate("0123456789abcdefghijklmnopqrstuvwxyz"):
        for index in range(5):
            sample_id = f"synthetic_{key}_{index:03d}"
            t = np.arange(14400) / 44100
            x = (
                0.08
                * np.sin(2 * np.pi * (400 + label * 90) * t)
                * np.exp(-(((t - 0.1) / 0.05) ** 2))
                + 0.002 * rng.standard_normal(len(t))
            ).astype(np.float32)
            path = processed / "clips" / (sample_id + ".wav")
            wavfile.write(path, 44100, x)
            rows.append(
                {
                    "sample_id": sample_id,
                    "recording_id": key,
                    "key": key,
                    "path": path.relative_to(processed).as_posix(),
                    "start_sample": index * 20000,
                    "end_sample": index * 20000 + 14400,
                    "sha256": sha(path),
                    "keep": 1,
                }
            )
    pd.DataFrame(rows).to_csv(processed / "manifest.csv", index=False)


def inspect_bundle(path, scope):
    """Check exported evidence, partition boundaries and the ZIP checksum."""
    with zipfile.ZipFile(path) as bundle:
        assert bundle.testzip() is None
        metadata = json.loads(bundle.read("bundle_metadata.json"))
        assert metadata["export_scope"] == scope and metadata["master_seed"] == 26
        assert not any(name.endswith(".pt") for name in bundle.namelist())
        assert "training/training_curves.png" in bundle.namelist()
        assert "source_snapshot.zip" in bundle.namelist()
        if scope == "main":
            assert not metadata["contains_validation_diagnostics"]
            assert not any(
                "exploratory_validation" in name for name in bundle.namelist()
            )
            assert "results/main_test/results_table.tex" in bundle.namelist()
        if scope in ["validation", "all"]:
            assert "results/exploratory_validation/predictions.csv" in bundle.namelist()


def main():
    """Exercise fresh preparation, resume, independent main export and exploration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source", choices=["synthetic", "public", "archive"], default="synthetic"
    )
    parser.add_argument("--raw-archive", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        work = Path(tempfile.mkdtemp(prefix="rsm_SOFTWARE_TEST_"))
    print("SOFTWARE TEST DIRECTORY:", work, flush=True)

    # Check that all stochastic repetition is reproducible under master seed 26.
    streams = [
        derived_seed(26, "gaussian", replicate, "test_sample") for replicate in range(3)
    ]
    assert len(set(streams)) == 3
    assert streams == [
        derived_seed(26, "gaussian", replicate, "test_sample") for replicate in range(3)
    ]
    assert paired_ci(np.zeros(6), np.array([0, 0, 0, 1, 1, 1]), reps=100, seed=26) == (
        0.0,
        0.0,
    )

    run = own_run(work / "SOFTWARE_TEST_run")
    data = local_data_root(work / "cache", run)
    if args.data_source == "synthetic":
        make_synthetic_data(data / "processed")
    else:
        if args.data_source == "archive" and args.raw_archive is None:
            raise ValueError("--raw-archive is required for archive mode")
        prepare(
            run,
            data,
            args.raw_archive.resolve() if args.data_source == "archive" else None,
        )
    processed = data / "processed"
    config = json.loads((ROOT / "configs/run.json").read_text())
    config.update(
        model="tiny_cnn",
        epochs=3,
        patience=0,
        baseline_description="Alternative small CNN, intended for pipeline smoke tests",
    )
    config_path = work / "tiny_config.json"
    config_path.write_text(json.dumps(config))
    review = work / "SIMULATED_QC_FIXTURE.json"
    review.write_text(
        json.dumps({"reviewer": "", "notes": "", "reviewed_sample_ids": []})
    )
    revision = args.data_source == "synthetic"
    expect_rejection(lambda: freeze(run, data, config_path, review, revision))
    review.write_text(
        json.dumps(
            {
                "reviewer": "SOFTWARE TEST FIXTURE ONLY",
                "notes": "Simulated acceptance for code testing. No human listening is claimed. Not research results.",
                "reviewed_sample_ids": qc_sample_ids(processed),
                "rejected": {},
            }
        )
    )
    freeze(run, data, config_path, review, revision)
    protocol = json.loads((run / "frozen_protocol.json").read_text())
    assert protocol["study"]["master_seed"] == protocol["training_config"]["seed"] == 26
    if not revision:
        assert (
            protocol["matches_reference_dataset"]
            and len(pd.read_csv(processed / "manifest.csv")) == 901
        )
    protected = snapshot({"data": processed})
    expect_rejection(lambda: freeze_checkpoint(run))
    unrelated = work / "UNRELATED_RESULTS"
    unrelated.mkdir()
    (unrelated / "preserve.txt").write_text("Do not overwrite unrelated outputs")
    unrelated_before = snapshot({"unrelated": unrelated})
    expect_rejection(lambda: own_run(unrelated))

    def stage(name, *extra):
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "rsm.pipeline",
                name,
                "--run",
                str(run),
                "--data-root",
                str(data),
                "--device",
                "cpu",
                *map(str, extra),
            ],
            cwd=ROOT,
            check=True,
        )

    # A pause preserves the full scheduler horizon; incomplete training cannot be tested.
    stage("train", "--session-epochs", 1)
    assert not (run / "training/training_complete.json").exists()
    expect_rejection(lambda: freeze_checkpoint(run))
    stage("train")
    uninterrupted = work / "uninterrupted"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "rsm.train",
            "--processed",
            str(processed),
            "--config",
            str(config_path),
            "--run",
            str(uninterrupted),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )
    a = torch.load(run / "training/last.pt", map_location="cpu", weights_only=True)
    b = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
    assert a["epoch"] == b["epoch"] == 3
    assert all(
        torch.equal(a["model_state"][key], value)
        for key, value in b["model_state"].items()
    )
    for row_a, row_b in zip(a["history"], b["history"]):
        assert {k: v for k, v in row_a.items() if k != "seconds"} == {
            k: v for k, v in row_b.items() if k != "seconds"
        }

    # Main export must work before exploratory validation exists.
    stage("sanity")
    stage("main")
    assert latest_completed(run, "exploratory_validation") is None
    stage("export-main")
    main_zip = next((run / "exports").glob("*_MAIN_results_*.zip"))
    inspect_bundle(main_zip, "main")
    main_output = latest_completed(run, "main_test")
    main_before = snapshot({"main": main_output, "archive": main_zip})

    # Exploration must preserve the completed main results.
    stage("diagnostics")
    stage("export-validation")
    stage("export-all")
    inspect_bundle(
        next((run / "exports").glob("*_VALIDATION_diagnostics_*.zip")), "validation"
    )
    inspect_bundle(next((run / "exports").glob("*_ALL_report_bundle_*.zip")), "all")
    assert main_before == snapshot({"main": main_output, "archive": main_zip})
    checkpoint_hash = sha(run / "training/best.pt")
    counts = {}
    for partition, name, conditions in [
        ("test", "main_test", 13),
        ("val", "exploratory_validation", 29),
    ]:
        output = latest_completed(run, name)
        meta = json.loads((output / "evaluation_metadata.json").read_text())
        df = pd.read_csv(output / "predictions.csv")
        assert meta["partition"] == partition and len(df) == meta["n"] * conditions
        assert (
            meta["checkpoint_sha256"] == checkpoint_hash
            and (df.master_seed == 26).all()
        )
        assert set(df[df.condition == "gaussian"].replicate_id) == {0, 1, 2}
        assert json.loads((output / "preservation_check.json").read_text())["passed"]
        assert (
            json.loads((output / "snr_audit.json").read_text())["max_target_error_db"]
            <= 0.1
        )
        counts[partition] = len(df)
    assert protected == snapshot({"data": processed})
    assert unrelated_before == snapshot({"unrelated": unrelated})
    results_before = snapshot({"results": run / "results"})
    stage("main")
    stage("diagnostics")
    stage("train")
    assert results_before == snapshot({"results": run / "results"})
    assert sha(run / "training/best.pt") == checkpoint_hash
    restored = local_data_root(work / "restored_cache", run)
    _, restored_processed = verify_frozen(run, restored)
    assert protected == snapshot({"data": restored_processed})

    # Invalid ZIP paths and changed completed outputs must not be accepted.
    bad = work / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../outside.txt", "invalid")
    expect_rejection(lambda: safe_extract(bad, work / "bad_extract"))
    assert not (work / "outside.txt").exists()
    from rsm.evaluation import reuse

    target = main_output / "metrics_summary.csv"
    content = target.read_bytes()
    target.write_bytes(content + b"\n")
    expect_rejection(lambda: reuse(run, "main_test", checkpoint_hash))
    target.write_bytes(content)
    result = {
        "passed": True,
        "scope": "SOFTWARE TEST ONLY: tiny CNN, three epochs, simulated QC",
        "data_source": args.data_source,
        "master_seed": 26,
        "matches_reference_dataset": protocol["matches_reference_dataset"],
        "main_rows": counts["test"],
        "validation_rows": counts["val"],
        "main_export_before_validation": True,
        "main_outputs_unchanged_by_validation": True,
        "resume_matches_uninterrupted": True,
        "restored_dataset_identical": True,
        "inputs_preserved": True,
        "completed_results_reused": True,
    }
    save_json(work / "SOFTWARE_VERIFICATION.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
