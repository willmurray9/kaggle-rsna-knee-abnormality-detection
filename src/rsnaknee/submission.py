import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Require the sample's ordered study IDs and valid probabilities for all targets."""
    for name, frame in (("Sample", sample), ("Submission", submission)):
        if list(frame.columns) != [ID_COLUMN, *TARGET_COLUMNS]:
            raise ValueError(f"{name} columns must match the official submission order.")
        if frame.empty:
            raise ValueError(f"{name} must not be empty.")
        ids = frame[ID_COLUMN]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError(f"{name} IDs must be unique, non-null, and nonblank.")
    if not submission[ID_COLUMN].reset_index(drop=True).equals(sample[ID_COLUMN].reset_index(drop=True)):
        raise ValueError("Submission IDs must exactly match the sample IDs in order.")
    probabilities = submission[TARGET_COLUMNS]
    if any(not is_numeric_dtype(dtype) or is_complex_dtype(dtype) for dtype in probabilities.dtypes):
        raise ValueError("Submission probabilities must be numeric values in [0, 1].")
    values = probabilities.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Submission probabilities must be finite values in [0, 1].")


def make_constant_submission(
    sample: pd.DataFrame, probabilities: pd.Series | None = None
) -> pd.DataFrame:
    """Build a constant baseline while preserving the official sample's study order."""
    if probabilities is None:
        probabilities = pd.Series(0.5, index=TARGET_COLUMNS)
    if list(probabilities.index) != TARGET_COLUMNS:
        raise ValueError("Probability index must match TARGET_COLUMNS in order.")
    submission = sample.copy()
    for target in TARGET_COLUMNS:
        submission[target] = probabilities[target]
    validate_submission(submission, sample)
    return submission
