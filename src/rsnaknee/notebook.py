"""Package the metadata model as one portable, offline Kaggle notebook."""

import argparse
import inspect
import json
from pathlib import Path

from rsnaknee.baseline import FEATURE_CATEGORIES, build_features, predict_metadata, predict_model
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256


def build_notebook(model_path: Path, output_path: Path) -> None:
    model = json.loads(model_path.read_text())
    inference = "\n\n".join(
        inspect.getsource(function) for function in (build_features, predict_model, predict_metadata)
    )
    definitions = (
        "from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\n\n"
        f"ID_COLUMN = {ID_COLUMN!r}\nTARGET_COLUMNS = {TARGET_COLUMNS!r}\n"
        f"FEATURE_CATEGORIES = {FEATURE_CATEGORIES!r}\n"
        f"MODEL = json.loads({json.dumps(model)!r})\n"
        f"MODEL_SHA256 = {sha256(model_path)!r}\n"
    )
    runtime = '''roots = [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
    Path("data/raw"), Path("../data/raw"),
]
data_root = next((p for p in roots if (p / "test.csv").is_file()), None)
if data_root is None:
    raise FileNotFoundError("Attach the RSNA knee competition data.")
test = pd.read_csv(data_root / "test.csv")
series = pd.read_csv(data_root / "test_series.csv")
sample = pd.read_csv(data_root / "sample_submission.csv")
if list(test.columns) != [ID_COLUMN] or test.empty:
    raise ValueError("Invalid test schema or no test studies")
if list(sample.columns) != [ID_COLUMN, *TARGET_COLUMNS]:
    raise ValueError("Unexpected sample submission schema")
submission = predict_metadata(test, series, MODEL)
if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
    raise ValueError("Output IDs differ from runtime test IDs")
values = submission[TARGET_COLUMNS].to_numpy()
if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
    raise ValueError("Invalid output probabilities")
output_dir = Path("/kaggle/working")
if not output_dir.is_dir():
    output_dir = data_root.resolve().parent.parent / "artifacts/baselines/notebook_metadata"
output_dir.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_dir / "submission.csv", index=False)
print(f"Wrote {len(submission)} studies; metadata model SHA256 {MODEL_SHA256}")
print(f"NumPy {np.__version__}; pandas {pd.__version__}; no training data needed")
'''
    cells = []
    for kind, source in [
        ("markdown", "# RSNA knee: metadata baseline\n\n"
         "Regularized logistic models use acquisition-series counts only. "
         "Trained on the 58 explicitly labeled studies; no report-derived labels, "
         "external data or image model. Validation is provisional study/report-grouped "
         "and does not establish patient independence.\n\n"
         "Generated from the repository's tested inference functions and a saved JSON model. "
         "Keep private, attach competition data, select CPU and disable internet. "
         "Hidden test IDs and series metadata are read at runtime.\n"),
        ("code", definitions), ("code", inference), ("code", runtime),
    ]:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_notebook(args.model, args.output)
    print(args.output)
