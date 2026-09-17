import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.eda import freeze_folds, report_groups, validate_folds


@pytest.fixture
def train():
    return pd.DataFrame(
        {
            ID_COLUMN: [f"study-{i:02d}" for i in range(30)],
            "Report": [f"Synthetic report {i}" for i in range(30)],
            **{target: [0, 1] * 15 for target in TARGET_COLUMNS},
        }
    )


def test_report_groups_normalize_text_without_grouping_missing_reports(train):
    train.loc[0, "Report"] = "  SYNTHETIC\nＲＥＰＯＲＴ  "
    train.loc[1, "Report"] = "synthetic report"
    train.loc[2, "Report"] = None
    train.loc[3, "Report"] = "  "

    groups = report_groups(train)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[2] != groups.iloc[3]
    assert groups.nunique() == len(train) - 1


def test_frozen_split_is_stable_grouped_and_does_not_modify_unknown_labels(train, tmp_path):
    train.loc[29, "Report"] = train.loc[0, "Report"]
    train.loc[27:, TARGET_COLUMNS] = np.nan
    original = train.copy(deep=True)

    folds = freeze_folds(train, tmp_path, {"train.csv": "hash"})
    saved_bytes = (tmp_path / "folds.csv").read_bytes()
    repeated = freeze_folds(train.sample(frac=1, random_state=8), tmp_path, {"train.csv": "hash"})

    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(folds, repeated)
    assert (tmp_path / "folds.csv").read_bytes() == saved_bytes
    assert folds.groupby("group_id").fold.nunique().max() == 1
    assert set(folds[ID_COLUMN]) == set(train[ID_COLUMN])
    assert validate_folds(train, folds)[["positive", "negative"]].gt(0).all().all()


def test_frozen_split_refuses_changed_input_instead_of_replacing_it(train, tmp_path):
    freeze_folds(train, tmp_path, {"train.csv": "original"})
    saved = (tmp_path / "folds.csv").read_bytes()

    with pytest.raises(ValueError, match="metadata changed"):
        freeze_folds(train, tmp_path, {"train.csv": "changed"})

    assert (tmp_path / "folds.csv").read_bytes() == saved


def test_frozen_split_rejects_edited_file(train, tmp_path):
    freeze_folds(train, tmp_path, {"train.csv": "hash"})
    path = tmp_path / "folds.csv"
    path.write_text(path.read_text() + "\n")

    with pytest.raises(ValueError, match="hash changed"):
        freeze_folds(train, tmp_path, {"train.csv": "hash"})


def test_split_rejects_a_study_dropped_or_duplicate_group_leakage(train, tmp_path):
    train.loc[29, "Report"] = train.loc[0, "Report"]
    folds = freeze_folds(train, tmp_path, {"train.csv": "hash"})
    with pytest.raises(ValueError, match="exactly once"):
        validate_folds(train, folds.iloc[1:])
    folds.loc[folds[ID_COLUMN].eq("study-29"), "fold"] = (
        int(folds.loc[folds[ID_COLUMN].eq("study-00"), "fold"].iloc[0]) + 1
    ) % 3
    with pytest.raises(ValueError, match="cross validation folds"):
        validate_folds(train, folds)


def test_split_rejects_undefined_target_auc(train, tmp_path):
    folds = freeze_folds(train, tmp_path, {"train.csv": "hash"})
    train.loc[train[ID_COLUMN].isin(folds.loc[folds.fold.eq(0), ID_COLUMN]), "ACL"] = 0
    with pytest.raises(ValueError, match="both classes.*ACL"):
        validate_folds(train, folds)
