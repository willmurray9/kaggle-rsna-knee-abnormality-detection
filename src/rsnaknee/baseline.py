"""A small acquisition-metadata baseline; no reports, images or IDs are predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from rsnaknee.constants import ID_COLUMN, METADATA_FILES, TARGET_COLUMNS
from rsnaknee.data import read_metadata, sha256
from rsnaknee.submission import validate_submission

SEED = 20260916
FEATURE_CATEGORIES = {
    "Anatomical_Plane": ("Axial", "Coronal", "Sagittal"),
    "Fluid_Sensitive": (0, 1),
    "Fat_Suppression": (0, 1),
}


def build_features(studies: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Fixed series counts in requested study order, including zero-series studies."""
    ids = pd.Index(studies[ID_COLUMN], name=ID_COLUMN)
    if ids.hasnans or ids.has_duplicates or (ids.astype(str).str.strip() == "").any():
        raise ValueError("Study IDs must be non-null, nonblank and unique")
    features = pd.DataFrame(index=ids)
    features["series_count"] = series.groupby(ID_COLUMN).size().reindex(ids, fill_value=0)
    for column, values in FEATURE_CATEGORIES.items():
        for value in values:
            counts = series.loc[series[column].eq(value)].groupby(ID_COLUMN).size()
            features[f"{column}={value}"] = counts.reindex(ids, fill_value=0)
    return features.astype(float)


def fit_model(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Fit only observed training labels; serialize portable scaler and linear weights."""
    if not features.index.equals(labels.index) or list(labels.columns) != TARGET_COLUMNS:
        raise ValueError("Features and labels must align, with all targets in official order")
    if not (labels.isna() | labels.isin([0, 1])).all().all():
        raise ValueError("Observed labels must be binary; unknown labels must remain missing")
    observed = labels.notna().any(axis=1)
    features, labels = features.loc[observed], labels.loc[observed]
    if features.empty or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Training features must be nonempty and finite")
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    coefficients, intercepts, counts = [], [], {}
    for target in TARGET_COLUMNS:
        known = labels[target].notna()
        if labels.loc[known, target].nunique() != 2:
            raise ValueError(f"Training labels require both classes for {target}")
        estimator = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        estimator.fit(scaled[known], labels.loc[known, target])
        coefficients.append(estimator.coef_[0].tolist())
        intercepts.append(float(estimator.intercept_[0]))
        counts[target] = int(known.sum())
    return {
        "feature_names": features.columns.tolist(),
        "targets": TARGET_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": coefficients,
        "intercepts": intercepts,
        "observed_counts": counts,
    }


def predict_model(model: dict, features: pd.DataFrame) -> pd.DataFrame:
    """Predict from plain JSON parameters using only NumPy and pandas."""
    if features.columns.tolist() != model["feature_names"] or model["targets"] != TARGET_COLUMNS:
        raise ValueError("Model feature and target order must match")
    scaled = (features.to_numpy(dtype=float) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = scaled @ np.asarray(model["coefficients"]).T + np.asarray(model["intercepts"])
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions must be finite")
    return pd.DataFrame(probabilities, index=features.index, columns=TARGET_COLUMNS)


def predict_metadata(studies: pd.DataFrame, series: pd.DataFrame, model: dict) -> pd.DataFrame:
    return predict_model(model, build_features(studies, series)).reset_index()


def _score(labels: pd.DataFrame, probabilities: pd.DataFrame) -> dict:
    scores = {}
    for target in TARGET_COLUMNS:
        known = labels[target].notna()
        if labels.loc[known, target].nunique() != 2:
            raise ValueError(f"Validation labels require both classes for {target}")
        scores[target] = float(roc_auc_score(labels.loc[known, target], probabilities.loc[known, target]))
    return {"macro_auc": float(np.mean(list(scores.values()))), "per_label_auc": scores}


def cross_validate(features: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame) -> dict:
    """Compare references and learned heads on the supplied, immutable group split."""
    if not features.index.equals(labels.index):
        raise ValueError("Features and labels must align")
    if folds[[ID_COLUMN, "group_id", "fold"]].isna().any().any() or folds[ID_COLUMN].duplicated().any():
        raise ValueError("Each study needs one nonmissing group and fold")
    if set(folds[ID_COLUMN]) != set(features.index):
        raise ValueError("Split must cover every training study exactly once")
    if (folds.groupby("group_id")["fold"].nunique() != 1).any():
        raise ValueError("A group must never span validation folds")
    split = folds.set_index(ID_COLUMN).reindex(features.index)
    fold_ids = sorted(split["fold"].unique())
    if len(fold_ids) < 2:
        raise ValueError("At least two validation folds are required")
    observed = labels.notna().any(axis=1)
    predictions = {
        name: pd.DataFrame(np.nan, index=labels.index[observed], columns=TARGET_COLUMNS)
        for name in ("constant", "prevalence", "learned")
    }
    models, scores = {}, []
    for fold in fold_ids:
        held_out = split["fold"].eq(fold)
        train = ~held_out & observed
        valid = held_out & observed
        model = fit_model(features.loc[train], labels.loc[train])
        models[str(fold)] = model
        predictions["learned"].loc[labels.index[valid]] = predict_model(model, features.loc[valid])
        predictions["constant"].loc[labels.index[valid]] = 0.5
        predictions["prevalence"].loc[labels.index[valid]] = labels.loc[train].mean().to_numpy()
        label_counts = {
            target: {
                "train_observed": int(labels.loc[train, target].notna().sum()),
                "train_positive": int(labels.loc[train, target].sum()),
                "valid_observed": int(labels.loc[valid, target].notna().sum()),
                "valid_positive": int(labels.loc[valid, target].sum()),
            }
            for target in TARGET_COLUMNS
        }
        scores.append({
            "fold": int(fold),
            "training_studies": int(train.sum()),
            "validation_studies": int(valid.sum()),
            "labels": label_counts,
            "scores": {
                name: _score(labels.loc[valid], prediction.loc[labels.index[valid]])
                for name, prediction in predictions.items()
            },
        })
    summary = {}
    for name in predictions:
        macro = [fold["scores"][name]["macro_auc"] for fold in scores]
        summary[name] = {
            "mean_macro_auc": float(np.mean(macro)),
            "std_macro_auc": float(np.std(macro, ddof=1)),
            "mean_per_label_auc": {
                target: float(np.mean([fold["scores"][name]["per_label_auc"][target] for fold in scores]))
                for target in TARGET_COLUMNS
            },
        }
    return {"models": models, "predictions": predictions, "folds": scores, "summary": summary}


def bootstrap_comparison(
    labels: pd.DataFrame, learned: pd.DataFrame, assignments: pd.Series, samples: int = 1000,
) -> dict:
    """Exploratory paired study bootstrap within folds, with the fitted OOF models fixed."""
    assignments = assignments.reindex(labels.index)
    reference = pd.DataFrame(0.5, index=labels.index, columns=TARGET_COLUMNS)
    differences = []
    rng = np.random.default_rng(SEED)
    partitions = [np.flatnonzero(assignments.eq(fold)) for fold in sorted(assignments.unique())]
    point = np.mean([
        _score(labels.iloc[rows], learned.iloc[rows])["macro_auc"] - 0.5
        for rows in partitions
    ])
    for _ in range(samples):
        fold_differences = []
        for rows in partitions:
            sampled = rng.choice(rows, size=len(rows), replace=True)
            truth = labels.iloc[sampled].reset_index(drop=True)
            if any(truth[target].dropna().nunique() != 2 for target in TARGET_COLUMNS):
                break
            candidate = learned.iloc[sampled].reset_index(drop=True)
            baseline = reference.iloc[sampled].reset_index(drop=True)
            fold_differences.append(_score(truth, candidate)["macro_auc"] - _score(truth, baseline)["macro_auc"])
        if len(fold_differences) == len(partitions):
            differences.append(float(np.mean(fold_differences)))
    interval = np.percentile(differences, [2.5, 97.5]).tolist() if differences else None
    return {
        "comparison": "Learned minus constant/prevalence mean within-fold macro AUC",
        "mean_auc_difference": float(point),
        "percentile_95_interval": interval,
        "requested_replicates": samples,
        "valid_replicates": len(differences),
        "skipped_single_class_replicates": samples - len(differences),
        "seed": SEED,
        "interpretation": (
            "Exploratory only: fixed OOF models, paired study resampling within each fold; "
            "omits model-fitting/split uncertainty and unresolved patient dependence. "
            "Replicates without both classes for every target in every fold are excluded."
        ),
    }


def run_baseline(raw_dir: Path, folds_path: Path, output: Path, bootstrap_samples: int = 1000) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"Preserve earlier experiments; output already exists: {output}")
    tables = read_metadata(raw_dir)
    input_hashes = {name: sha256(raw_dir / name) for name in METADATA_FILES}
    split_hash = sha256(folds_path)
    split_manifest_path = folds_path.with_name("folds_manifest.json")
    split_manifest = json.loads(split_manifest_path.read_text()) if split_manifest_path.exists() else None
    if split_manifest and (
        split_manifest["folds_sha256"] != split_hash or split_manifest["input_sha256"] != input_hashes
    ):
        raise ValueError("Split manifest and current input/split hashes differ")
    train = tables["train.csv"]
    labels = train.set_index(ID_COLUMN)[TARGET_COLUMNS]
    features = build_features(train, tables["train_series.csv"])
    folds = pd.read_csv(folds_path)
    result = cross_validate(features, labels, folds)
    observed = labels.notna().any(axis=1)
    labeled = labels.loc[observed]
    bootstrap = bootstrap_comparison(
        labeled, result["predictions"]["learned"], folds.set_index(ID_COLUMN)["fold"], bootstrap_samples,
    )
    model = fit_model(features, labels)
    submission = predict_metadata(tables["sample_submission.csv"], tables["test_series.csv"], model)
    validate_submission(submission, tables["sample_submission.csv"])

    output.mkdir(parents=True)
    features.to_csv(output / "train_features.csv")
    labeled.to_csv(output / "observed_labels.csv")
    oof = folds.set_index(ID_COLUMN).loc[labeled.index].copy()
    for target in TARGET_COLUMNS:
        oof[f"observed_{target}"] = labeled[target]
        for name, predictions in result["predictions"].items():
            oof[f"{name}_{target}"] = predictions[target]
    oof["mean_squared_probability_error"] = (labeled - result["predictions"]["learned"]).pow(2).mean(axis=1)
    oof.to_csv(output / "oof.csv")
    oof.sort_values("mean_squared_probability_error", ascending=False).head(10).to_csv(output / "largest_errors.csv")
    submission.to_csv(output / "submission.csv", index=False)
    (output / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (output / "fold_models.json").write_text(json.dumps(result["models"], indent=2) + "\n")
    shutil.copy2(folds_path, output / "folds.csv")
    shutil.copy2(Path(__file__), output / "baseline_source.py")
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=root, capture_output=True).stdout
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Acquisition metadata may contain weak disease signal; diagnostic baseline only",
        "training_labeled_studies": int(observed.sum()),
        "excluded_unlabeled_studies": int((~observed).sum()),
        "label_provenance": "Observed train.csv labels only; missing labels excluded",
        "features": model["feature_names"],
        "feature_policy": "Fixed per-study counts; unrecognized categories contribute only to total series count",
        "estimator": {"type": "StandardScaler + per-target LogisticRegression", "C": 0.1, "max_iter": 1000, "seed": SEED, "class_weight": None},
        "preprocessing": "Scaler fitted to observed training studies within each fold, then all observed studies for final fit",
        "weight_provenance": "Locally fitted from competition training metadata; no external weights",
        "cv": result["summary"],
        "folds": result["folds"],
        "bootstrap": bootstrap,
        "metric_note": "Equal-weight mean of fold AUCs; pooled prevalence OOF AUC is deliberately not reported",
        "selected_for_submission": bool(result["summary"]["learned"]["mean_macro_auc"] > 0.5),
        "selection_rule": "One prespecified candidate: use learned model if mean fold macro AUC exceeds 0.5; otherwise constant smoke model",
        "limitations": ["Only 58 observed studies in the competition snapshot; substantial validation uncertainty", "Patient overlap and image duplicates remain unresolved", "Acquisition metadata may encode protocol/site shortcuts; no image understanding", "Fluid_Sensitive and Fat_Suppression are redundant in current metadata"],
        "input_sha256": input_hashes,
        "split_sha256": split_hash,
        "split_manifest": split_manifest,
        "code": {"revision": revision, "dirty": bool(status), "status": status, "diff_sha256": hashlib.sha256(diff).hexdigest()},
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in sorted((root / "src" / "rsnaknee").glob("*.py"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}},
        "artifact_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds", type=Path, default=Path("data/processed/folds.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments") / datetime.now(timezone.utc).strftime("metadata-%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    summary = run_baseline(args.raw_dir, args.folds, args.output)
    print(json.dumps({"output": str(args.output), "cv": summary["cv"], "bootstrap": summary["bootstrap"], "runtime_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
