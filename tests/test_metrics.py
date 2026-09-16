import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import TARGET_COLUMNS
from rsnaknee.metrics import macro_auc


@pytest.fixture
def labels_and_predictions():
    labels = pd.DataFrame({target: [0, 0, 1, 1] for target in TARGET_COLUMNS})
    predictions = pd.DataFrame(
        {target: [0.1, 0.2, 0.8, 0.9] for target in TARGET_COLUMNS}
    )
    return labels, predictions


def test_macro_auc_weights_each_label_equally(labels_and_predictions):
    labels, predictions = labels_and_predictions
    predictions["ACL"] = [0.9, 0.8, 0.2, 0.1]
    result = macro_auc(labels, predictions)
    assert result["macro_auc"] == pytest.approx(11 / 12)
    assert result["per_label_auc"]["ACL"] == 0.0
    assert result["per_label_auc"]["MCL"] == 1.0
    assert macro_auc(labels, predictions * 0 + 0.5)["macro_auc"] == 0.5


def test_macro_auc_rejects_misaligned_rows_or_columns(labels_and_predictions):
    labels, predictions = labels_and_predictions
    with pytest.raises(ValueError, match="index"):
        macro_auc(labels, predictions.iloc[::-1])
    with pytest.raises(ValueError, match="columns"):
        macro_auc(labels, predictions[TARGET_COLUMNS[::-1]])
    with pytest.raises(ValueError, match="columns"):
        macro_auc(labels.drop(columns="ACL"), predictions)
    with pytest.raises(ValueError, match="empty"):
        macro_auc(labels.iloc[:0], predictions.iloc[:0])


@pytest.mark.parametrize("value", [np.nan, 2, "1"])
def test_macro_auc_rejects_incomplete_or_nonbinary_labels(
    labels_and_predictions, value
):
    labels, predictions = labels_and_predictions
    labels["ACL"] = labels["ACL"].astype(object)
    labels.loc[0, "ACL"] = value
    with pytest.raises(ValueError, match="binary"):
        macro_auc(labels, predictions)


def test_macro_auc_identifies_single_class_targets(labels_and_predictions):
    labels, predictions = labels_and_predictions
    labels["ACL"] = 0
    with pytest.raises(ValueError, match="ACL"):
        macro_auc(labels, predictions)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1, "0.5"])
def test_macro_auc_rejects_invalid_probabilities(labels_and_predictions, value):
    labels, predictions = labels_and_predictions
    if isinstance(value, str):
        predictions["ACL"] = predictions["ACL"].astype(object)
    predictions.loc[0, "ACL"] = value
    with pytest.raises(ValueError, match="probabilities"):
        macro_auc(labels, predictions)
