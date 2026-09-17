import json

import numpy as np
import pandas as pd
import pytest

from rsnaknee.baseline import (
    bootstrap_comparison, build_features, cross_validate, fit_model,
    predict_metadata, predict_model, run_baseline,
)
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS


def test_features_count_series_and_preserve_requested_study_order() -> None:
    studies = pd.DataFrame({ID_COLUMN: ["b", "a", "missing"]})
    series = pd.DataFrame({
        ID_COLUMN: ["a", "a", "b"],
        "Anatomical_Plane": ["Axial", "Sagittal", "Coronal"],
        "Fluid_Sensitive": [0, 1, 1],
        "Fat_Suppression": [0, 1, 0],
    })

    features = build_features(studies, series)

    assert features.index.tolist() == ["b", "a", "missing"]
    np.testing.assert_array_equal(features.to_numpy(), [
        [1, 0, 1, 0, 0, 1, 1, 0],
        [2, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])


def test_unknown_labels_are_excluded_and_inputs_are_preserved() -> None:
    features = pd.DataFrame({"count": [0, 1, 2, 3]}, index=list("abcd"))
    labels = pd.DataFrame({target: [0, 0, 1, 1] for target in TARGET_COLUMNS}, index=features.index)
    labels.loc["b", "ACL"] = np.nan
    expected = fit_model(features, labels)
    features.loc["unknown"] = 1_000_000
    labels.loc["unknown"] = np.nan
    original = labels.copy(deep=True)

    actual = fit_model(features, labels)

    pd.testing.assert_frame_equal(labels, original)
    assert actual == expected
    assert actual["observed_counts"]["ACL"] == 3
    assert actual["observed_counts"]["MCL"] == 4


def test_json_inference_produces_all_targets_with_aligned_ids() -> None:
    studies = pd.DataFrame({ID_COLUMN: ["b", "a", "c", "d"]})
    series = pd.DataFrame({
        ID_COLUMN: ["a", "b", "c", "d"],
        "Anatomical_Plane": ["Axial", "Sagittal", "Coronal", "Axial"],
        "Fluid_Sensitive": [0, 1, 0, 1],
        "Fat_Suppression": [0, 1, 0, 1],
    })
    features = build_features(studies, series)
    labels = pd.DataFrame({target: [0, 1, 0, 1] for target in TARGET_COLUMNS}, index=features.index)
    model = json.loads(json.dumps(fit_model(features, labels)))

    predictions = predict_metadata(studies, series.sample(frac=1, random_state=2), model)

    assert predictions.columns.tolist() == [ID_COLUMN, *TARGET_COLUMNS]
    assert predictions[ID_COLUMN].tolist() == ["b", "a", "c", "d"]
    assert np.isfinite(predictions[TARGET_COLUMNS]).all().all()
    assert predictions[TARGET_COLUMNS].ge(0).all().all()
    assert predictions[TARGET_COLUMNS].le(1).all().all()
    pd.testing.assert_frame_equal(
        predictions.set_index(ID_COLUMN), predict_model(model, features)
    )


def test_json_scaling_and_distinct_target_coefficients_match_known_probabilities() -> None:
    model = {
        "feature_names": ["count"], "targets": TARGET_COLUMNS,
        "mean": [5.0], "scale": [2.0],
        "coefficients": [[weight] for weight in range(-6, 6)],
        "intercepts": [0.0] * 12,
    }

    probabilities = predict_model(model, pd.DataFrame({"count": [7.0]}, index=["study-a"]))

    np.testing.assert_allclose(probabilities.iloc[0], [
        0.002472623, 0.006692851, 0.017986210, 0.047425873,
        0.119202922, 0.268941421, 0.5, 0.731058579,
        0.880797078, 0.952574127, 0.982013790, 0.993307149,
    ], atol=1e-8)


def cv_data():
    ids = list("abcdefgh")
    features = pd.DataFrame({"count": [0, 1, 2, 3, 100, 101, 102, 103]}, index=ids)
    labels = pd.DataFrame({target: [0, 1, 0, 1, 0, 1, 0, 1] for target in TARGET_COLUMNS}, index=ids)
    folds = pd.DataFrame({ID_COLUMN: ids, "group_id": ids, "fold": [0, 0, 0, 0, 1, 1, 1, 1]})
    return features, labels, folds


def test_cv_model_and_scaler_use_only_training_partition() -> None:
    features, labels, folds = cv_data()

    result = cross_validate(features, labels, folds)

    assert result["models"]["0"]["mean"] == [101.5]
    assert result["models"]["1"]["mean"] == [1.5]
    changed = features.copy()
    changed.loc[list("abcd"), "count"] += 10_000
    repeated = cross_validate(changed, labels, folds)
    assert result["models"]["0"] == repeated["models"]["0"]
    assert result["summary"]["constant"]["mean_macro_auc"] == 0.5
    assert result["summary"]["prevalence"]["mean_macro_auc"] == 0.5
    assert result["predictions"]["learned"].index.tolist() == list("abcdefgh")


def test_cv_rejects_groups_spanning_folds() -> None:
    features, labels, folds = cv_data()
    folds.loc[4, "group_id"] = "a"

    with pytest.raises(ValueError, match="group"):
        cross_validate(features, labels, folds)


def test_cv_refuses_undefined_auc_instead_of_dropping_target() -> None:
    features, labels, folds = cv_data()
    labels.loc[list("abcd"), "Fracture"] = 0

    with pytest.raises(ValueError, match="Fracture"):
        cross_validate(features, labels, folds)


def test_paired_bootstrap_is_reproducible_and_reports_skipped_draws() -> None:
    features, labels, folds = cv_data()
    predictions = cross_validate(features, labels, folds)["predictions"]
    assignments = folds.set_index(ID_COLUMN)["fold"]

    result = bootstrap_comparison(labels, predictions["learned"], assignments, samples=50)

    assert result == bootstrap_comparison(labels, predictions["learned"], assignments, samples=50)
    assert result["valid_replicates"] + result["skipped_single_class_replicates"] == 50
    assert result["valid_replicates"] > 0
    assert result["mean_auc_difference"] == 0.25
    assert result["percentile_95_interval"][0] <= 0.25 <= result["percentile_95_interval"][1]


def test_experiment_saves_portable_model_and_refuses_overwrite(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _, labels, folds = cv_data()
    train = labels.rename_axis(ID_COLUMN).reset_index()
    train.insert(1, "Report", "training supervision only")
    train.to_csv(raw / "train.csv", index=False)
    test = pd.DataFrame({ID_COLUMN: ["test-b", "test-a"]})
    test.to_csv(raw / "test.csv", index=False)
    sample = test.assign(**{target: 0.5 for target in TARGET_COLUMNS})
    sample.to_csv(raw / "sample_submission.csv", index=False)
    for name, studies in (("train", train), ("test", test)):
        series = pd.DataFrame({
            ID_COLUMN: studies[ID_COLUMN],
            "SeriesInstanceUID": [f"series-{uid}" for uid in studies[ID_COLUMN]],
            "Fluid_Sensitive": 0,
            "Fat_Suppression": 0,
            "Anatomical_Plane": "Axial",
        })
        series.to_csv(raw / f"{name}_series.csv", index=False)
    split_path = tmp_path / "folds.csv"
    folds.to_csv(split_path, index=False)
    output = tmp_path / "run"

    summary = run_baseline(raw, split_path, output, bootstrap_samples=10)

    saved_model = json.loads((output / "model.json").read_text())
    submission = pd.read_csv(output / "submission.csv")
    recomputed = predict_metadata(test, pd.read_csv(raw / "test_series.csv"), saved_model)
    pd.testing.assert_frame_equal(submission, recomputed)
    assert summary["training_labeled_studies"] == 8
    assert summary["label_provenance"] == "Observed train.csv labels only; missing labels excluded"
    assert summary["artifact_sha256"]["model.json"]
    assert summary["split_sha256"]
    assert len(pd.read_csv(output / "oof.csv")) == 8
    with pytest.raises(FileExistsError):
        run_baseline(raw, split_path, output, bootstrap_samples=10)
