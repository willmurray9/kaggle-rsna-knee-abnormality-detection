from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import audit_metadata, read_metadata


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    train = pd.DataFrame(
        {
            ID_COLUMN: ["train-a", "train-b", "train-c"],
            "Report": ["Synthetic report A", "Synthetic report B", ""],
            **{target: [1.0, np.nan, np.nan] for target in TARGET_COLUMNS},
        }
    )
    train.loc[1, "ACL"] = 0.0
    test = pd.DataFrame({ID_COLUMN: ["test-a", "test-b"]})
    tables = {"train.csv": train, "test.csv": test}
    for split, studies in (("train", train), ("test", test)):
        tables[f"{split}_series.csv"] = pd.DataFrame(
            {
                ID_COLUMN: studies[ID_COLUMN],
                "SeriesInstanceUID": [f"{uid}-series" for uid in studies[ID_COLUMN]],
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
        )
    tables["sample_submission.csv"] = test.assign(**{target: 0.5 for target in TARGET_COLUMNS})
    for name, frame in tables.items():
        frame.to_csv(tmp_path / name, index=False)
    return tmp_path


def test_missing_labels_remain_unknown(metadata_dir: Path) -> None:
    train = read_metadata(metadata_dir)["train.csv"]

    assert train.loc[0, TARGET_COLUMNS].eq(1).all()
    assert train.loc[1, "ACL"] == 0
    assert train.loc[1, TARGET_COLUMNS[1:]].isna().all()
    assert train.loc[2, TARGET_COLUMNS].isna().all()


def test_audit_counts_observed_and_missing_labels(metadata_dir: Path) -> None:
    report = audit_metadata(
        {"raw_dir": metadata_dir, "artifacts_dir": metadata_dir / "artifacts", "competition": "synthetic"}
    )

    assert report["fully_labeled_studies"] == 1
    assert report["partially_labeled_studies"] == 1
    assert report["unlabeled_studies"] == 1
    assert report["nonempty_reports"] == 2
    assert report["labels"]["ACL"] == {"observed": 2, "positive": 1}
    assert report["labels"]["MCL"] == {"observed": 1, "positive": 1}


@pytest.mark.parametrize("label", [-1, 2, 0.5, "unknown"])
def test_invalid_labels_are_rejected(metadata_dir: Path, label: object) -> None:
    train = pd.read_csv(metadata_dir / "train.csv")
    train["ACL"] = train["ACL"].astype(object)
    train.loc[0, "ACL"] = label
    train.to_csv(metadata_dir / "train.csv", index=False)

    with pytest.raises(ValueError, match="Observed labels must be binary"):
        read_metadata(metadata_dir)


def test_train_test_study_overlap_is_rejected(metadata_dir: Path) -> None:
    test = pd.read_csv(metadata_dir / "test.csv")
    test.loc[0, ID_COLUMN] = "train-a"
    test.to_csv(metadata_dir / "test.csv", index=False)

    with pytest.raises(ValueError, match="Training and test studies overlap"):
        read_metadata(metadata_dir)


@pytest.mark.parametrize("split", ["train", "test"])
def test_studies_without_series_are_rejected(metadata_dir: Path, split: str) -> None:
    path = metadata_dir / f"{split}_series.csv"
    series = pd.read_csv(path)
    series.iloc[1:].to_csv(path, index=False)

    with pytest.raises(ValueError, match=f"{split} studies and series metadata do not match"):
        read_metadata(metadata_dir)


@pytest.mark.parametrize("split", ["train", "test"])
def test_orphan_series_are_rejected(metadata_dir: Path, split: str) -> None:
    path = metadata_dir / f"{split}_series.csv"
    series = pd.read_csv(path)
    orphan = series.iloc[[0]].copy()
    orphan[ID_COLUMN] = "unknown-study"
    orphan["SeriesInstanceUID"] = "orphan-series"
    pd.concat([series, orphan], ignore_index=True).to_csv(path, index=False)

    with pytest.raises(ValueError, match=f"{split} studies and series metadata do not match"):
        read_metadata(metadata_dir)


@pytest.mark.parametrize("filename", ["train.csv", "test.csv", "train_series.csv", "test_series.csv"])
def test_unexpected_schema_is_rejected(metadata_dir: Path, filename: str) -> None:
    path = metadata_dir / filename
    frame = pd.read_csv(path)
    frame["unexpected_column"] = 1
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Unexpected .* schema"):
        read_metadata(metadata_dir)


def test_missing_metadata_file_is_rejected(metadata_dir: Path) -> None:
    (metadata_dir / "test_series.csv").unlink()

    with pytest.raises(FileNotFoundError, match="Missing metadata: .*test_series.csv"):
        read_metadata(metadata_dir)
