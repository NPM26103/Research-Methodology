"""Pinned raw recordings -> candidates -> declared manual QC -> frozen data."""

import concurrent.futures, hashlib, json, shutil, time, urllib.request, zipfile
from pathlib import Path
import pandas as pd
from .io import (
    ROOT,
    call,
    archive_folder,
    now,
    source_snapshot,
    source_hashes,
    new_attempt,
)
from .safety import sha, save_json, safe_extract
from .audit import audit_data


def restore_archive(archive, target):
    """Restore into a staging folder so an interruption does not expose partial data."""
    target = Path(target)
    if target.exists():
        return
    staging = target.with_name(target.name + ".restoring")
    if staging.exists():
        shutil.rmtree(staging)
    safe_extract(archive, staging)
    staging.replace(target)


def restore_data(run, data):
    """Verify the persisted dataset ZIP before restoring the local runtime cache."""
    record = run / "data_archive.json"
    if not record.exists():
        return False
    info = json.loads(record.read_text())
    archive = run / info["path"]
    if sha(archive) != info["sha256"]:
        raise ValueError("Persisted data archive changed")
    restore_archive(archive, Path(data) / "processed")
    return True


def prepare(run, data, workspace_zip=None):
    """Download verified raw data, segment candidates and persist them before manual QC."""
    data = Path(data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    if restore_data(run, data):
        print("Restored/cached frozen data; no repeated preprocessing.", flush=True)
        return
    prepared = run / "artifacts/prepared_dataset.zip"
    record = run / "prepared_archive.json"
    if record.exists():
        if sha(prepared) != json.loads(record.read_text())["sha256"]:
            raise ValueError("Prepared archive changed")
        restore_archive(prepared, data / "processed")
        return
    # A completed ZIP can outlive a disconnect just before its record was saved.
    if prepared.exists():
        with zipfile.ZipFile(prepared) as z:
            if (
                z.testzip() is not None
                or "preparation_complete.json" not in z.namelist()
            ):
                raise ValueError("Incomplete prepared archive")
        restore_archive(prepared, data / "processed")
        save_json(record, {"sha256": sha(prepared), "created_utc": now()})
        return
    reference = json.loads((ROOT / "configs/raw_reference.json").read_text())
    raw = data / "raw"
    raw.mkdir(exist_ok=True)
    archive = zipfile.ZipFile(workspace_zip) if workspace_zip else None
    try:

        def fetch(entry):
            dest = raw / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if sha(dest) != entry["sha256"]:
                    raise ValueError("Raw file changed: " + str(dest))
                return
            if archive is not None:
                content = archive.read("data/raw/" + entry["path"])
            else:
                url = f"https://raw.githubusercontent.com/JBFH-Dev/Keystroke-Datasets/{reference['commit']}/{entry['path']}"
                # A new Colab session downloads public data directly; no old
                # dataset, checkpoint or result archive is required.
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(
                            urllib.request.Request(
                                url,
                                headers={"User-Agent": "RsM-research-data-downloader"},
                            ),
                            timeout=60,
                        ) as response:
                            content = response.read()
                        break
                    except (OSError, TimeoutError):
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Raw SHA-256 mismatch: " + entry["path"])
            temp = dest.with_suffix(dest.suffix + ".tmp")
            temp.write_bytes(content)
            temp.replace(dest)
            print("Verified raw:", entry["path"], flush=True)

        if archive is not None:
            for entry in reference["files"]:
                fetch(entry)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(fetch, reference["files"]))
    finally:
        if archive is not None:
            archive.close()
    (raw / "SOURCE.json").write_text(json.dumps(reference, indent=2))
    call(run, "rsm.audit_raw", "--source", raw / "MBPWavs", "--out", data / "raw_audit")
    processed = data / "processed"
    if not (processed / "preparation_complete.json").exists():
        if processed.exists():
            shutil.rmtree(processed)  # only an unfinished, owned local cache
        call(
            run,
            "rsm.prepare_data",
            "--source",
            raw / "MBPWavs",
            "--out",
            processed,
            "--settings",
            ROOT / "configs/segmentation_overrides.json",
        )
        save_json(
            processed / "preparation_complete.json",
            {"complete": True, "created_utc": now()},
        )
    artifacts = run / "artifacts"
    artifacts.mkdir(exist_ok=True)
    shutil.copy2(data / "raw_audit/raw_manifest.csv", artifacts / "raw_manifest.csv")
    shutil.copy2(raw / "SOURCE.json", artifacts / "raw_source.json")
    digest = archive_folder(processed, prepared)
    save_json(record, {"sha256": digest, "created_utc": now()})
    print(
        "Candidates persisted. Listen and record QC before freezing the split.",
        flush=True,
    )


def qc_sample_ids(processed):
    """Select one fixed clip per class plus every j candidate for minimum human QC."""
    m = pd.read_csv(
        Path(processed) / "manifest.csv", dtype={"sample_id": str, "key": str}
    )
    selected = set(m.sort_values("sample_id").groupby("key").first().sample_id)
    selected.update(m.loc[m.key == "j", "sample_id"])
    return sorted(selected)


def freeze(run, data, config, qc_review, allow_revision=False):
    """Validate the human QC declaration, split once, audit and lock data/config."""
    processed = Path(data).resolve() / "processed"
    lock = run / "frozen_protocol.json"
    if lock.exists():
        verify_frozen(run, data)
        if (
            json.loads(Path(config).read_text())
            != json.loads(lock.read_text())["training_config"]
        ):
            raise ValueError("Frozen training config changed")
        print("Keeping frozen data and config.")
        return
    review = json.loads(Path(qc_review).read_text())
    if (
        not str(review.get("reviewer", "")).strip()
        or not str(review.get("notes", "")).strip()
    ):
        raise ValueError("Enter actual reviewer and audio QC notes")
    m = pd.read_csv(processed / "manifest.csv", dtype={"sample_id": str, "key": str})
    reviewed = set(review.get("reviewed_sample_ids", []))
    if not reviewed.issubset(set(m.sample_id)) or not set(
        qc_sample_ids(processed)
    ).issubset(reviewed):
        raise ValueError(
            "Record actual listening: at least one clip/class and every j candidate"
        )
    rejected = review.get("rejected", {})
    if (
        not isinstance(rejected, dict)
        or not set(rejected).issubset(reviewed)
        or any(not str(v).strip() for v in rejected.values())
    ):
        raise ValueError(
            "Rejected samples need audio-based reasons and must have been reviewed"
        )
    if rejected and not allow_revision:
        raise ValueError(
            "Data revision: use a new run with --allow-data-revision and document rejected samples"
        )
    if (processed / "split.csv").exists():
        saved = processed / "qc_review.json"
        if not saved.exists() or json.loads(saved.read_text()) != review:
            raise ValueError("QC differs after split creation")
    else:
        if rejected:
            m.loc[m.sample_id.isin(rejected), "keep"] = 0
            m.to_csv(processed / "manifest.csv", index=False)
        (processed / "qc_review.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False)
        )
        call(run, "rsm.split_data", "--processed", processed, "--seed", 26)
    hashes = {
        k + "_sha256": sha(processed / f)
        for k, f in [
            ("manifest", "manifest.csv"),
            ("split", "split.csv"),
            ("label_map", "label_map.json"),
        ]
    }
    matches = hashes == json.loads(
        (ROOT / "configs/verified_data_hashes.json").read_text()
    )
    if not matches and not allow_revision:
        raise ValueError(
            "Dataset differs from the supplied integrity reference; inspect before accepting a data revision"
        )
    cfg = json.loads(Path(config).read_text())
    study = json.loads((ROOT / "configs/study.json").read_text())
    if not all(
        value == 26
        for value in [
            cfg["seed"],
            study["master_seed"],
            study["split_seed"],
            study["training_seed"],
            study["bootstrap_seed"],
        ]
    ):
        raise ValueError(
            "This submission protocol requires master/split/training/bootstrap seed 26"
        )
    s = pd.read_csv(processed / "split.csv")
    ck = {
        "data_hashes": hashes,
        "label_map": json.loads((processed / "label_map.json").read_text()),
        "config": cfg,
        "train_ids": s.loc[s.split == "train", "sample_id"].tolist(),
        "val_ids": s.loc[s.split == "val", "sample_id"].tolist(),
    }
    audit_data(processed, ck, new_attempt(run, "data_audit"))
    metadata = run / "data_metadata"
    metadata.mkdir(exist_ok=True)
    for p in processed.glob("*"):
        if p.is_file():
            shutil.copy2(p, metadata / p.name)
    (run / "training_config.json").write_text(json.dumps(cfg, indent=2))
    archive = run / "artifacts/processed_dataset.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Bad processed archive")
            for name in [
                "manifest.csv",
                "split.csv",
                "label_map.json",
                "qc_review.json",
            ]:
                if z.read(name) != (processed / name).read_bytes():
                    raise ValueError("Incomplete freeze archive differs")
        digest = sha(archive)
    else:
        digest = archive_folder(processed, archive)
    if not (run / "data_archive.json").exists():
        save_json(
            run / "data_archive.json",
            {"path": "artifacts/processed_dataset.zip", "sha256": digest},
        )
    if not (run / "source_snapshot.zip").exists():
        source_snapshot(run / "source_snapshot.zip")
    save_json(
        lock,
        {
            "study": study,
            "training_config": cfg,
            "data_hashes": hashes,
            "source_sha256": source_hashes(),
            "matches_reference_dataset": matches,
            "data_revision": not matches,
            "qc_review_sha256": sha(processed / "qc_review.json"),
            "created_utc": now(),
        },
    )
    print(
        "Frozen:",
        len(m[m.keep == 1]),
        "clips; split seed 26; training seed",
        cfg["seed"],
        flush=True,
    )


def verify_frozen(run, data):
    """Restore the cache if needed and reject changed code, data metadata or QC."""
    path = run / "frozen_protocol.json"
    if not path.exists():
        raise ValueError("Complete QC and freeze first")
    protocol = json.loads(path.read_text())
    if protocol["source_sha256"] != source_hashes():
        raise ValueError("Code changed after freeze")
    restore_data(run, data)
    root = Path(data).resolve() / "processed"
    for key, name in [
        ("manifest_sha256", "manifest.csv"),
        ("split_sha256", "split.csv"),
        ("label_map_sha256", "label_map.json"),
    ]:
        if sha(root / name) != protocol["data_hashes"][key]:
            raise ValueError("Frozen dataset metadata changed")
    if sha(root / "qc_review.json") != protocol["qc_review_sha256"]:
        raise ValueError("QC record changed")
    return protocol, root
