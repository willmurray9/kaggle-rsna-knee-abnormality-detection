import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.submission import make_constant_submission, validate_submission


@pytest.fixture
def sample():
    frame = pd.DataFrame(0.0, index=range(2), columns=TARGET_COLUMNS)
    frame.insert(0, ID_COLUMN, ["study-b", "study-a"])
    return frame


def test_constant_submission_preserves_ids_and_does_not_mutate_sample(sample):
    original = sample.copy()
    submission = make_constant_submission(sample)
    assert submission[ID_COLUMN].tolist() == ["study-b", "study-a"]
    assert (submission[TARGET_COLUMNS] == 0.5).all().all()
    pd.testing.assert_frame_equal(sample, original)
    validate_submission(submission, sample)


def test_constant_submission_uses_label_specific_probabilities(sample):
    probabilities = pd.Series(0.2, index=TARGET_COLUMNS)
    probabilities["ACL"] = 0.8
    submission = make_constant_submission(sample, probabilities)
    assert submission["ACL"].tolist() == [0.8, 0.8]
    assert submission["MCL"].tolist() == [0.2, 0.2]
    with pytest.raises(ValueError, match="index"):
        make_constant_submission(sample, probabilities.iloc[::-1])
    probabilities["ACL"] = 1.1
    with pytest.raises(ValueError, match="probabilities"):
        make_constant_submission(sample, probabilities)


def test_submission_rejects_wrong_row_order_and_column_order(sample):
    with pytest.raises(ValueError, match="IDs"):
        validate_submission(sample.iloc[::-1], sample)
    with pytest.raises(ValueError, match="columns"):
        validate_submission(sample[sample.columns[::-1]], sample)
    with pytest.raises(ValueError, match="columns"):
        validate_submission(sample, sample.drop(columns="ACL"))
    with pytest.raises(ValueError, match="empty"):
        validate_submission(sample.iloc[:0], sample.iloc[:0])


@pytest.mark.parametrize("ids", [["a", "a"], ["a", None], ["a", "  "]])
def test_submission_rejects_invalid_ids_even_when_sample_matches(sample, ids):
    sample[ID_COLUMN] = ids
    with pytest.raises(ValueError, match="IDs"):
        validate_submission(sample, sample)
    with pytest.raises(ValueError, match="IDs"):
        make_constant_submission(sample)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1, "0.5"])
def test_submission_rejects_invalid_probabilities(sample, value):
    submission = sample.copy()
    if isinstance(value, str):
        submission["ACL"] = submission["ACL"].astype(object)
    submission.loc[0, "ACL"] = value
    with pytest.raises(ValueError, match="probabilities"):
        validate_submission(submission, sample)
