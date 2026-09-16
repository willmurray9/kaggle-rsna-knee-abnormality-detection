from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rsnaknee.constants import ID_COLUMN, METADATA_FILES, TARGET_COLUMNS
from rsnaknee.submission import validate_submission


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_metadata(config: dict) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    root = config["raw_dir"]
    root.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    for name in METADATA_FILES:
        api.competition_download_file(config["competition"], name, path=str(root), quiet=True)
        archive = root / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as zipped:
                # Read only the requested member; never extract arbitrary paths.
                (root / name).write_bytes(zipped.read(name))
            archive.unlink()


def _check_ids(frame: pd.DataFrame, column: str, unique: bool = True) -> None:
    ids = frame[column]
    if frame.empty or ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{column} contains empty IDs or no rows")
    if unique and ids.duplicated().any():
        raise ValueError(f"{column} contains duplicate IDs")


def read_metadata(root: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in METADATA_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing metadata: {missing}. Run make download.")
    tables = {name: pd.read_csv(root / name) for name in METADATA_FILES}
    train, test = tables["train.csv"], tables["test.csv"]
    if list(train.columns) != [ID_COLUMN, "Report", *TARGET_COLUMNS]:
        raise ValueError("Unexpected train.csv schema; inspect the competition data before proceeding")
    if list(test.columns) != [ID_COLUMN]:
        raise ValueError("Unexpected test.csv schema")
    for frame in (train, test):
        _check_ids(frame, ID_COLUMN)
    if set(train[ID_COLUMN]) & set(test[ID_COLUMN]):
        raise ValueError("Training and test studies overlap")
    labels = train[TARGET_COLUMNS]
    if not (labels.isna() | labels.isin([0, 1])).all().all():
        raise ValueError("Observed labels must be binary; unknown labels must remain missing")
    sample = tables["sample_submission.csv"]
    validate_submission(sample, sample)
    if set(sample[ID_COLUMN]) != set(test[ID_COLUMN]):
        raise ValueError("Sample submission and test studies differ")
    series_columns = [ID_COLUMN, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
    for split, studies in (("train", train), ("test", test)):
        series = tables[f"{split}_series.csv"]
        if list(series.columns) != series_columns:
            raise ValueError(f"Unexpected {split}_series.csv schema")
        _check_ids(series, ID_COLUMN, unique=False)
        _check_ids(series, "SeriesInstanceUID")
        if set(series[ID_COLUMN]) != set(studies[ID_COLUMN]):
            raise ValueError(f"{split} studies and series metadata do not match")
    if set(tables["train_series.csv"]["SeriesInstanceUID"]) & set(tables["test_series.csv"]["SeriesInstanceUID"]):
        raise ValueError("Training and test series overlap")
    return tables


def audit_metadata(config: dict) -> dict:
    tables = read_metadata(config["raw_dir"])
    train = tables["train.csv"]
    labels = train[TARGET_COLUMNS]
    observed = labels.notna()
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition": config["competition"],
        "file_sha256": {name: sha256(config["raw_dir"] / name) for name in METADATA_FILES},
        "rows": {name: len(frame) for name, frame in tables.items()},
        "fully_labeled_studies": int(observed.all(axis=1).sum()),
        "partially_labeled_studies": int((observed.any(axis=1) & ~observed.all(axis=1)).sum()),
        "unlabeled_studies": int((~observed.any(axis=1)).sum()),
        "nonempty_reports": int(train["Report"].fillna("").str.strip().ne("").sum()),
        "labels": {
            name: {"observed": int(observed[name].sum()), "positive": int(labels[name].sum())}
            for name in TARGET_COLUMNS
        },
        "series_by_plane": tables["train_series.csv"]["Anatomical_Plane"].value_counts().to_dict(),
        "patient_id_available_in_csv": False,
    }
    out = config["artifacts_dir"] / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Metadata audit", "",
        f"Training studies: {len(train):,}; training series: {len(tables['train_series.csv']):,}.",
        f"Complete labels: {report['fully_labeled_studies']}; partial labels: {report['partially_labeled_studies']}; no labels: {report['unlabeled_studies']}.",
        "", "Missing labels are unknown, not negative. Reports are available only in training.",
        "CSV metadata has study IDs but no patient IDs. Study separation alone does not establish patient independence.",
        "", "| Target | Observed | Positive |", "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {name} | {row['observed']} | {row['positive']} |" for name, row in report["labels"].items())
    (out / "data_audit.md").write_text("\n".join(lines) + "\n")
    return report
