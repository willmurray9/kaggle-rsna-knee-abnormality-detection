import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype
from sklearn.metrics import roc_auc_score

from rsnaknee.constants import TARGET_COLUMNS


def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    """Score fully labeled, aligned studies; never silently omit a target."""
    if list(y_true.columns) != TARGET_COLUMNS or list(y_pred.columns) != TARGET_COLUMNS:
        raise ValueError("Label and prediction columns must match TARGET_COLUMNS in order.")
    if y_true.empty or y_pred.empty:
        raise ValueError("Cannot score empty labels or predictions.")
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Label and prediction index must match in order.")
    if not y_true.isin([0, 1]).to_numpy().all():
        raise ValueError("Labels must be complete binary values (0 or 1).")
    single_class = [target for target in TARGET_COLUMNS if y_true[target].nunique() < 2]
    if single_class:
        raise ValueError(f"ROC AUC requires both classes for every target: {single_class}")
    if any(not is_numeric_dtype(dtype) or is_complex_dtype(dtype) for dtype in y_pred.dtypes):
        raise ValueError("Predictions must be numeric probabilities in [0, 1].")
    values = y_pred.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    scores = {
        target: float(roc_auc_score(y_true[target], y_pred[target]))
        for target in TARGET_COLUMNS
    }
    return {"macro_auc": float(np.mean(list(scores.values()))), "per_label_auc": scores}
