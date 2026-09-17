import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import TARGET_COLUMNS
from rsnaknee.image_audit import audit_images


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_manifests(features: Path, folds: Path, train: Path) -> None:
    train_hash = hashlib.sha256(train.read_bytes()).hexdigest()
    (features / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "artifact_sha256": {name: hashlib.sha256((features / name).read_bytes()).hexdigest()
                            for name in ("IDs.csv", "quality.csv", "selected_series.csv")},
        "input_sha256": {"train.csv": train_hash},
    }))
    folds.with_name("folds_manifest.json").write_text(json.dumps({
        "folds_sha256": hashlib.sha256(folds.read_bytes()).hexdigest(),
        "input_sha256": {"train.csv": train_hash},
    }))


@pytest.fixture
def audit_inputs(tmp_path: Path):
    features = tmp_path / "features"
    features.mkdir()
    ids = pd.DataFrame({"StudyInstanceUID": ["gold-a", "gold-b", "silver-c", "test-d"],
                        "split": ["train", "train", "train", "test"]})
    ids.to_csv(features / "IDs.csv", index=False)
    ids.assign(usable_planes=3, missing_planes="", header_failures=0, decode_failures=0,
               repeated_failed_samples=0, spacing_fallbacks=0, instance_number_planes=0).to_csv(
        features / "quality.csv", index=False,
    )
    rows = []
    for study, split in ids.itertuples(index=False, name=None):
        for plane in ["Sagittal", "Coronal", "Axial"]:
            rows.append({
                "StudyInstanceUID": study, "split": split, "plane": plane, "status": "usable",
                "patient_id_sha256": digest("shared patient") if study != "gold-b" else digest("other patient"),
                "patient_key_sha256": digest("shared key") if study != "gold-b" else digest("other key"),
                "issuer_sha256": digest("issuer"), "site_sha256": digest("site"),
                "scanner_sha256": digest("scanner"), "patient_identity_removed": "YES",
                "sample_pixel_sha256": json.dumps([digest("shared pixels"), digest("shared pixels"), None])
                if study in {"gold-a", "gold-b", "test-d"} else "[]",
            })
    pd.DataFrame(rows).to_csv(features / "selected_series.csv", index=False)
    train = tmp_path / "train.csv"
    pd.DataFrame({"StudyInstanceUID": ["gold-a", "gold-b", "silver-c"],
                  "Report": ["private raw report"] * 3,
                  **{target: [0, 1, np.nan] for target in TARGET_COLUMNS}}).to_csv(train, index=False)
    folds = tmp_path / "folds.csv"
    pd.DataFrame({"StudyInstanceUID": ["gold-a", "gold-b", "silver-c"],
                  "group_id": ["a", "b", "c"], "fold": [0, 1, 1]}).to_csv(folds, index=False)
    write_manifests(features, folds, train)
    return features, folds, train


def test_patient_overlap_counts_cross_fold_gold_and_train_test_without_identifiers(audit_inputs) -> None:
    report = audit_images(*audit_inputs)
    patient = report["overlap"]["patient_key_sha256"]
    assert patient["unique_keys"] == 2
    assert patient["multi_study_groups"] == 1
    assert patient["cross_fold_groups"] == 1
    assert patient["cross_fold_groups_touching_gold"] == 1
    assert patient["cross_fold_gold_to_gold_groups"] == 0
    assert patient["cross_fold_gold_studies"] == 1
    assert patient["train_test_groups"] == 1
    assert patient["train_test_gold_studies"] == 1
    serialized = json.dumps(report)
    for private in ["gold-a", "private raw report", digest("shared patient"), digest("shared key")]:
        assert private not in serialized


def test_pixel_duplicates_deduplicate_repeated_samples_and_series_within_study(audit_inputs) -> None:
    report = audit_images(*audit_inputs)
    pixels = report["overlap"]["sample_pixel_sha256"]
    assert pixels["unique_keys"] == 1
    assert pixels["largest_group_studies"] == 3
    assert pixels["cross_fold_gold_to_gold_groups"] == 1
    assert pixels["cross_fold_gold_studies"] == 2
    assert pixels["train_test_groups"] == 1
    assert "not exhaustive" in " ".join(report["limitations"])


def test_empty_hashes_and_unknown_labels_do_not_create_groups_or_gold(audit_inputs) -> None:
    features, _, _ = audit_inputs
    rows = pd.read_csv(features / "selected_series.csv").fillna("")
    rows["patient_id_sha256"] = ""
    rows["patient_key_sha256"] = ""
    rows.to_csv(features / "selected_series.csv", index=False)
    write_manifests(*audit_inputs)
    report = audit_images(*audit_inputs)
    assert report["overlap"]["patient_key_sha256"]["unique_keys"] == 0
    assert report["studies"]["fully_observed_gold"] == 2
    assert report["studies"]["unlabeled_train"] == 1


def test_missing_fold_is_rejected(audit_inputs) -> None:
    _, folds, _ = audit_inputs
    pd.read_csv(folds).iloc[:2].to_csv(folds, index=False)
    with pytest.raises(ValueError, match="fold"):
        audit_images(*audit_inputs)


def test_incomplete_selected_plane_audit_is_rejected(audit_inputs) -> None:
    features, _, _ = audit_inputs
    pd.read_csv(features / "selected_series.csv").iloc[1:].to_csv(features / "selected_series.csv", index=False)
    write_manifests(*audit_inputs)
    with pytest.raises(ValueError, match="plane"):
        audit_images(*audit_inputs)


def test_manifest_checksum_mismatch_is_rejected(audit_inputs) -> None:
    features, _, _ = audit_inputs
    (features / "manifest.json").write_text(json.dumps({
        "status": "complete", "artifact_sha256": {"selected_series.csv": "incorrect"},
    }))
    with pytest.raises(ValueError, match="hash"):
        audit_images(*audit_inputs)


def test_multiple_patient_keys_within_study_are_counted(audit_inputs) -> None:
    features, _, _ = audit_inputs
    rows = pd.read_csv(features / "selected_series.csv")
    rows.loc[0, "patient_key_sha256"] = digest("different within-study key")
    rows.to_csv(features / "selected_series.csv", index=False)
    write_manifests(*audit_inputs)
    assert audit_images(*audit_inputs)["overlap"]["patient_key_sha256"]["studies_with_multiple_keys"] == 1


def test_failed_extraction_and_malformed_pixel_hashes_are_rejected(audit_inputs) -> None:
    features, _, _ = audit_inputs
    (features / "manifest.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="complete"):
        audit_images(*audit_inputs)
    (features / "manifest.json").write_text('{"status":"complete"}')
    rows = pd.read_csv(features / "selected_series.csv")
    rows.loc[0, "sample_pixel_sha256"] = '["not a valid digest"]'
    rows.to_csv(features / "selected_series.csv", index=False)
    write_manifests(*audit_inputs)
    with pytest.raises(ValueError, match="hash"):
        audit_images(*audit_inputs)


@pytest.mark.parametrize("section,name", [
    ("artifact_sha256", "IDs.csv"), ("artifact_sha256", "quality.csv"),
    ("artifact_sha256", "selected_series.csv"), ("input_sha256", "train.csv"),
])
def test_each_feature_manifest_hash_is_required(audit_inputs, section: str, name: str) -> None:
    features, _, _ = audit_inputs
    path = features / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest[section][name]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        audit_images(*audit_inputs)


def test_edited_fold_assignment_with_same_studies_is_rejected(audit_inputs) -> None:
    _, folds, _ = audit_inputs
    changed = pd.read_csv(folds)
    changed.loc[0, "fold"] = 1
    changed.to_csv(folds, index=False)
    with pytest.raises(ValueError, match="fold.*hash|hash.*fold"):
        audit_images(*audit_inputs)


def test_folds_manifest_is_required(audit_inputs) -> None:
    _, folds, _ = audit_inputs
    folds.with_name("folds_manifest.json").unlink()
    with pytest.raises(ValueError, match="fold.*manifest"):
        audit_images(*audit_inputs)


@pytest.mark.parametrize("field", ["folds_sha256", "train.csv"])
def test_folds_manifest_hash_fields_are_required_and_matched(audit_inputs, field: str) -> None:
    _, folds, _ = audit_inputs
    path = folds.with_name("folds_manifest.json")
    original = json.loads(path.read_text())
    for value in (None, "incorrect"):
        manifest = json.loads(json.dumps(original))
        target = manifest if field == "folds_sha256" else manifest["input_sha256"]
        if value is None:
            del target[field]
        else:
            target[field] = value
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="hash"):
            audit_images(*audit_inputs)
