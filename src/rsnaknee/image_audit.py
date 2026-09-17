"""Aggregate identifier and sampled-pixel overlap without exporting member identifiers."""

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256


HASH_COLUMNS = (
    "patient_key_sha256", "patient_id_sha256", "issuer_sha256", "site_sha256", "scanner_sha256",
)


def _overlap(
    pairs: list[tuple[str, str]],
    split_by_study: dict[str, str],
    fold_by_study: dict[str, int],
    gold: set[str],
) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    study_keys: dict[str, set[str]] = {}
    for study, key in pairs:
        if not key:
            continue
        if re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Invalid identifier or pixel hash")
        groups.setdefault(key, set()).add(study)
        study_keys.setdefault(study, set()).add(key)
    cross_fold, train_test = [], []
    gold_to_gold = 0
    for studies in groups.values():
        train = {study for study in studies if split_by_study[study] == "train"}
        if len({fold_by_study[study] for study in train}) > 1:
            cross_fold.append(train)
        if len({fold_by_study[study] for study in train & gold}) > 1:
            gold_to_gold += 1
        if train and len(train) < len(studies):
            train_test.append(studies)
    crossing_studies = set().union(*cross_fold) if cross_fold else set()
    train_test_studies = set().union(*train_test) if train_test else set()
    return {
        "studies_with_key": len(study_keys),
        "unique_keys": len(groups),
        "multi_study_groups": sum(len(studies) > 1 for studies in groups.values()),
        "largest_group_studies": max(map(len, groups.values()), default=0),
        "studies_with_multiple_keys": sum(len(keys) > 1 for keys in study_keys.values()),
        "cross_fold_groups": len(cross_fold),
        "cross_fold_train_studies": len(crossing_studies),
        "cross_fold_groups_touching_gold": sum(bool(studies & gold) for studies in cross_fold),
        "cross_fold_gold_to_gold_groups": gold_to_gold,
        "cross_fold_gold_studies": len(crossing_studies & gold),
        "train_test_groups": len(train_test),
        "train_test_studies": len(train_test_studies),
        "train_test_gold_studies": len(train_test_studies & gold),
        "per_fold_studies_with_key": {
            str(fold): sum(study in study_keys for study, value in fold_by_study.items() if value == fold)
            for fold in sorted(set(fold_by_study.values()))
        },
    }


def audit_images(features_dir: Path, folds_path: Path, train_path: Path) -> dict[str, Any]:
    """Audit complete extraction outputs against unchanged folds and observed labels.

    Reads only study IDs and target columns from train.csv, never Report. Returned
    groups are summarized as counts; no member study IDs or identifier hashes leave
    this function. Cross-fold counts refer to the supplied saved split, not a new fit.
    """
    features_dir, folds_path, train_path = map(Path, (features_dir, folds_path, train_path))
    manifest_path = features_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Image audit requires a complete extraction")
    artifact_names = ("IDs.csv", "quality.csv", "selected_series.csv", "manifest.json")
    input_hashes = {
        **{name: sha256(features_dir / name) for name in artifact_names},
        "folds.csv": sha256(folds_path), "train.csv": sha256(train_path),
    }
    for name in ("IDs.csv", "quality.csv", "selected_series.csv"):
        if manifest.get("artifact_sha256", {}).get(name) != input_hashes[name]:
            raise ValueError(f"Required artifact hash missing or differs from extraction manifest: {name}")
    if manifest.get("input_sha256", {}).get("train.csv") != input_hashes["train.csv"]:
        raise ValueError("Required train.csv hash missing or differs from extraction manifest")
    folds_manifest_path = folds_path.with_name("folds_manifest.json")
    if not folds_manifest_path.is_file():
        raise ValueError("Saved folds manifest is required beside folds.csv")
    folds_manifest = json.loads(folds_manifest_path.read_text())
    if folds_manifest.get("folds_sha256") != input_hashes["folds.csv"]:
        raise ValueError("Required folds.csv hash missing or differs from saved folds manifest")
    if folds_manifest.get("input_sha256", {}).get("train.csv") != input_hashes["train.csv"]:
        raise ValueError("Required train.csv hash missing or differs from saved folds manifest")
    input_hashes["folds_manifest.json"] = sha256(folds_manifest_path)
    ids = pd.read_csv(features_dir / "IDs.csv", dtype=str, keep_default_na=False)
    quality = pd.read_csv(features_dir / "quality.csv", dtype=str, keep_default_na=False)
    selected = pd.read_csv(features_dir / "selected_series.csv", dtype=str, keep_default_na=False)
    train = pd.read_csv(train_path, usecols=[ID_COLUMN, *TARGET_COLUMNS], dtype={ID_COLUMN: str})
    folds = pd.read_csv(folds_path, dtype={ID_COLUMN: str})
    if ids.empty or ids[ID_COLUMN].eq("").any() or ids[ID_COLUMN].duplicated().any():
        raise ValueError("Invalid extraction study IDs")
    if not set(ids["split"]).issubset({"train", "test"}):
        raise ValueError("Invalid extraction split values")
    split_by_study = dict(zip(ids[ID_COLUMN], ids["split"]))
    expected_train = {study for study, split in split_by_study.items() if split == "train"}
    if train[ID_COLUMN].isna().any() or train[ID_COLUMN].duplicated().any() or set(train[ID_COLUMN]) != expected_train:
        raise ValueError("Training studies do not match feature IDs")
    if folds[ID_COLUMN].isna().any() or folds[ID_COLUMN].duplicated().any() or set(folds[ID_COLUMN]) != expected_train:
        raise ValueError("Saved folds must contain every training study exactly once")
    fold_values = pd.to_numeric(folds["fold"], errors="raise")
    if fold_values.isna().any() or (fold_values < 0).any() or (fold_values % 1 != 0).any():
        raise ValueError("Saved fold values must be nonnegative integers")
    fold_by_study = dict(zip(folds[ID_COLUMN], fold_values.astype(int)))
    for name, table in (("quality", quality), ("selected series", selected)):
        if set(table[ID_COLUMN]) != set(split_by_study):
            raise ValueError(f"{name} study IDs do not match extraction IDs")
        if not table["split"].eq(table[ID_COLUMN].map(split_by_study)).all():
            raise ValueError(f"{name} split values do not match extraction IDs")
    if quality[ID_COLUMN].duplicated().any() or selected.duplicated([ID_COLUMN, "plane"]).any():
        raise ValueError("Duplicate quality or study/plane audit rows")
    if not selected.groupby(ID_COLUMN)["plane"].agg(set).map(lambda value: value == {"Sagittal", "Coronal", "Axial"}).all():
        raise ValueError("Every study must have all three plane audit rows, including missing planes")
    labels = train[TARGET_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not (labels.isna() | labels.isin([0, 1])).all().all():
        raise ValueError("Observed labels must be binary")
    fully_observed = labels.notna().all(axis=1)
    gold = set(train.loc[fully_observed, ID_COLUMN])
    overlap = {
        column: _overlap(list(zip(selected[ID_COLUMN], selected[column])), split_by_study, fold_by_study, gold)
        for column in HASH_COLUMNS
    }
    pixel_pairs = []
    for study, raw in zip(selected[ID_COLUMN], selected["sample_pixel_sha256"]):
        values = json.loads(raw)
        if not isinstance(values, list) or any(value is not None and not isinstance(value, str) for value in values):
            raise ValueError("Invalid sampled pixel hash list")
        pixel_pairs.extend((study, value) for value in values if value)
    overlap["sample_pixel_sha256"] = _overlap(pixel_pairs, split_by_study, fold_by_study, gold)
    id_issuer = selected.loc[selected["patient_id_sha256"].ne("") & selected["issuer_sha256"].ne("")]
    totals = {}
    for key in ("header_failures", "decode_failures", "repeated_failed_samples", "spacing_fallbacks", "instance_number_planes"):
        values = pd.to_numeric(quality[key], errors="raise")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError("Invalid quality counts")
        totals[key] = int(values.sum())
    usable = pd.to_numeric(quality["usable_planes"], errors="raise")
    return {
        "patient_independence_established": False,
        "studies": {
            "total": len(ids), "train": len(train), "test": len(ids) - len(train),
            "fully_observed_gold": int(fully_observed.sum()),
            "partially_observed_train": int((labels.notna().any(axis=1) & ~fully_observed).sum()),
            "unlabeled_train": int(labels.isna().all(axis=1).sum()),
        },
        "quality": {
            **totals,
            "studies_with_missing_planes": int(usable.lt(3).sum()),
            "studies_with_zero_usable_planes": int(usable.eq(0).sum()),
            "series_status_counts": {str(k): int(v) for k, v in selected["status"].value_counts().items()},
        },
        "identifier_reliability": {
            "patient_keys_without_issuer_series": int((selected["patient_key_sha256"].ne("") & selected["issuer_sha256"].eq("")).sum()),
            "patient_id_groups_with_multiple_issuers": int(id_issuer.groupby("patient_id_sha256")["issuer_sha256"].nunique().gt(1).sum()),
            "patient_identity_removed_series": {str(k): int(v) for k, v in selected["patient_identity_removed"].value_counts().items()},
        },
        "overlap": overlap,
        "input_sha256": input_hashes,
        "definitions": {
            "gold": "All 12 target labels explicitly observed; unknown labels stay unknown.",
            "cross_fold": "A key shared by distinct training studies assigned different saved folds.",
            "groups_touching_gold": "Cross-fold key group contains at least one fully observed gold study.",
            "gold_to_gold": "Fully observed gold studies sharing a key occupy at least two different folds.",
            "sample_pixel_sha256": "Identical decoded, modality-rescaled, MONOCHROME-adjusted float32 pixel bytes among sampled slices.",
        },
        "limitations": [
            "Patient hashes preserve equality, not identity validity; anonymization may reuse placeholder IDs or assign new IDs per study.",
            "PatientID without a trustworthy issuer can collide across institutions; a combined patient key does not prove independence.",
            "Site and scanner keys reflect available header fields; shared keys across folds are expected and are not patient leakage evidence.",
            "Only selected series and sampled slices per plane are checked; this is not exhaustive duplicate or patient linkage detection.",
            "Pixel hashes omit array shape and spatial metadata; identical bytes, including constant images, are duplicate candidates rather than proof of duplicate studies.",
            "Identifier audit uses one selected header per series; deidentification flags do not establish reliable patient linkage.",
            "No fold assignments are changed; cross-fold matches require review before interpreting validation as patient independent.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_images(args.features, args.folds, args.train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
