import json
from pathlib import Path

import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.submission import validate_submission


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def execute_notebook(name: str | Path) -> None:
    notebook = json.loads((NOTEBOOKS / name).read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(name), "exec"), namespace)


def test_smoke_uses_replaced_test_ids_without_training_data(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    test = pd.DataFrame({ID_COLUMN: [f"hidden-{i}" for i in range(1300)]})
    test.to_csv(raw / "test.csv", index=False)
    # The example sample can retain its three original rows during hidden reruns.
    sample = pd.DataFrame({ID_COLUMN: ["example-a", "example-b", "example-c"]})
    sample[TARGET_COLUMNS] = 0.0
    sample.to_csv(raw / "sample_submission.csv", index=False)
    monkeypatch.chdir(tmp_path)

    execute_notebook("00-submission-smoke.ipynb")

    result = pd.read_csv(tmp_path / "artifacts/baselines/notebook_smoke/submission.csv")
    expected = test.copy()
    expected[TARGET_COLUMNS] = 0.5
    validate_submission(result, expected)
    pd.testing.assert_frame_equal(result, expected)


def test_smoke_rejects_duplicate_hidden_ids(tmp_path, monkeypatch):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    test = pd.DataFrame({ID_COLUMN: ["duplicate", "duplicate"]})
    test.to_csv(raw / "test.csv", index=False)
    sample = test.copy()
    sample[TARGET_COLUMNS] = 0.0
    sample.to_csv(raw / "sample_submission.csv", index=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Invalid test IDs"):
        execute_notebook("00-submission-smoke.ipynb")


def test_metadata_notebook_matches_model_on_hidden_ids_without_training_data(tmp_path, monkeypatch):
    from rsnaknee.baseline import build_features, predict_metadata
    from rsnaknee.notebook import build_notebook

    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    test = pd.DataFrame({ID_COLUMN: [f"hidden-{i}" for i in range(1300)]})
    series = pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN].iloc[1:],
        "SeriesInstanceUID": [f"series-{i}" for i in range(1299)],
        "Anatomical_Plane": "Sagittal",
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 0,
    })
    series.loc[series.index[-1], "Anatomical_Plane"] = "Unknown plane"
    features = build_features(test, series)
    n_features = len(features.columns)
    model = {
        "feature_names": features.columns.tolist(), "targets": TARGET_COLUMNS,
        "mean": [0.2] * n_features, "scale": [1.5] * n_features,
        "coefficients": [[0.01 * (i + 1)] * n_features for i in range(12)],
        "intercepts": [-0.5] * 12, "observed_counts": {},
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    notebook_path = tmp_path / "metadata.ipynb"
    build_notebook(model_path, notebook_path)
    model_path.unlink()  # Runtime uses embedded parameters, not local artifacts.
    test.to_csv(raw / "test.csv", index=False)
    series.to_csv(raw / "test_series.csv", index=False)
    sample = pd.DataFrame({ID_COLUMN: ["example-a", "example-b", "example-c"]})
    sample[TARGET_COLUMNS] = 0.0
    sample.to_csv(raw / "sample_submission.csv", index=False)
    monkeypatch.chdir(tmp_path)

    execute_notebook(notebook_path)

    result = pd.read_csv(tmp_path / "artifacts/baselines/notebook_metadata/submission.csv")
    expected = predict_metadata(test, series, model)
    validate_submission(result, expected)
    pd.testing.assert_frame_equal(result, expected)
