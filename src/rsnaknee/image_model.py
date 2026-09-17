"""Compare three prespecified supervision recipes on fixed image embeddings."""

import argparse
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rsnaknee.baseline import _score
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256
from rsnaknee.submission import validate_submission

RECIPES = {"observed_only": 0.0, "silver_full": 1.0, "silver_quarter": 0.25}
SEED = 20260916


def fit_heads(features: pd.DataFrame, labels: pd.DataFrame, *, silver_weight: float,
              pca_components: int | None = None) -> dict:
    if not features.index.equals(labels.index) or features.index.has_duplicates:
        raise ValueError("Features and labels must have unique aligned study indices")
    if not 0 <= silver_weight <= 1:
        raise ValueError("Silver weight must be in [0, 1]")
    gold = labels[[target + "__observed" for target in TARGET_COLUMNS]].copy()
    gold.columns = TARGET_COLUMNS
    silver = labels[TARGET_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        mask = labels[target + "__mask"]
        if not pd.api.types.is_bool_dtype(mask.dtype) or mask.isna().any():
            raise ValueError("Supervision masks must be nonmissing boolean values")
        silver[target] = silver[target].where(mask & labels[target + "__verdict"].isin(["YES", "NO"]))
    truth = gold.combine_first(silver) if silver_weight else gold
    if not (truth.isna() | truth.isin([0, 1])).all().all():
        raise ValueError("Observed and derived targets must be binary or unknown")
    usable = truth.notna().any(axis=1)
    x = features.loc[usable]
    if x.empty or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Training features must be finite and nonempty")
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    pca = None
    if pca_components is not None:
        if not isinstance(pca_components, int) or not 1 <= pca_components <= min(transformed.shape):
            raise ValueError("PCA components must fit the training matrix dimensions")
        pca = PCA(n_components=pca_components, whiten=False, svd_solver="randomized", random_state=SEED)
        pca.fit(transformed)
        transformed = pca.transform(transformed)
    coefficients, intercepts, counts = [], [], {}
    for target in TARGET_COLUMNS:
        target_truth = truth.loc[usable, target]
        known = target_truth.notna()
        observed = gold.loc[usable, target].notna()
        if target_truth[known].nunique() != 2:
            raise ValueError(f"Training target lacks both classes: {target}")
        weights = np.where(observed[known], 1.0, silver_weight)
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        model.fit(transformed[known], target_truth[known], sample_weight=weights)
        coefficient = model.coef_[0] if pca is None else model.coef_[0] @ pca.components_
        intercept = model.intercept_[0] if pca is None else model.intercept_[0] - coefficient @ pca.mean_
        coefficients.append(coefficient.tolist())
        intercepts.append(float(intercept))
        counts[target] = {"observed": int((known & observed).sum()), "derived": int((known & ~observed).sum())}
    result = {
        "feature_names": features.columns.tolist(), "targets": TARGET_COLUMNS,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": coefficients, "intercepts": intercepts,
        "counts": counts, "silver_weight": silver_weight,
    }
    if pca is not None:
        result["pca"] = {"components": pca_components, "whiten": False,
                         "svd_solver": "randomized", "seed": SEED, "training_rows": len(x),
                         "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                         "total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum())}
    return result


def predict_heads(model: dict, features: pd.DataFrame) -> pd.DataFrame:
    if features.columns.tolist() != model["feature_names"] or model["targets"] != TARGET_COLUMNS:
        raise ValueError("Model feature/target order differs")
    x = (features.to_numpy() - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = x @ np.asarray(model["coefficients"]).T + np.asarray(model["intercepts"])
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite image predictions")
    return pd.DataFrame(probabilities, index=features.index, columns=TARGET_COLUMNS)


def compare_supervision(features: pd.DataFrame, labels: pd.DataFrame, *,
                        pca_components: int | None = None) -> dict:
    if not features.index.equals(labels.index):
        raise ValueError("Features and labels must align")
    if labels[["fold", "group_id"]].isna().any().any():
        raise ValueError("Missing fold/group assignments")
    if labels.groupby("group_id")["fold"].nunique().gt(1).any():
        raise ValueError("A report group crosses validation folds")
    gold = labels[[target + "__observed" for target in TARGET_COLUMNS]].copy()
    gold.columns = TARGET_COLUMNS
    observed = gold.notna().any(axis=1)
    result = {}
    for recipe, silver_weight in RECIPES.items():
        oof = pd.DataFrame(np.nan, index=labels.index[observed], columns=TARGET_COLUMNS)
        models, scores = {}, []
        for fold in sorted(labels["fold"].unique()):
            training = labels["fold"].ne(fold)
            validation = ~training & observed
            model = fit_heads(features.loc[training], labels.loc[training], silver_weight=silver_weight,
                              pca_components=pca_components)
            predictions = predict_heads(model, features.loc[validation])
            oof.loc[predictions.index] = predictions
            models[str(fold)] = model
            scores.append({"fold": int(fold), **_score(gold.loc[validation], predictions)})
        macro = [row["macro_auc"] for row in scores]
        result[recipe] = {
            "models": models, "oof": oof, "folds": scores,
            "mean_macro_auc": float(np.mean(macro)),
            "std_macro_auc": float(np.std(macro, ddof=1)),
            "mean_per_target_auc": {
                target: float(np.mean([row["per_label_auc"][target] for row in scores]))
                for target in TARGET_COLUMNS
            },
        }
    return result


def read_frozen_labels(labels_path: Path, audit_path: Path) -> pd.DataFrame:
    audit = json.loads(audit_path.read_text())
    folds_path = labels_path.with_name("folds.csv")
    if sha256(labels_path) != audit["label_table_sha256"] or sha256(folds_path) != audit["folds_sha256"]:
        raise ValueError("Label table or frozen split hash differs from the label audit")
    labels = pd.read_csv(labels_path).set_index(ID_COLUMN)
    folds = pd.read_csv(folds_path).set_index(ID_COLUMN)
    if labels.index.has_duplicates or folds.index.has_duplicates or set(labels.index) != set(folds.index):
        raise ValueError("Label and frozen fold assignments have different studies")
    if not labels[["group_id", "fold"]].equals(folds.reindex(labels.index)[["group_id", "fold"]]):
        raise ValueError("Label table assignments differ from the frozen split")
    return labels


def read_image_features(features_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    manifest = json.loads((features_dir / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["encoder_fit_on_competition_data"]:
        raise ValueError("Need complete features from an encoder not fitted on competition data")
    for name in ("IDs.csv", "features.npy"):
        if sha256(features_dir / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Image cache hash differs: {name}")
    ids = pd.read_csv(features_dir / "IDs.csv")
    values = np.load(features_dir / "features.npy", allow_pickle=False)
    if (len(values) != len(ids) or list(values.shape) != manifest["feature_shape"]
            or manifest["completed_studies"] != len(ids) or ids[ID_COLUMN].isna().any()
            or ids[ID_COLUMN].duplicated().any() or not np.isfinite(values).all()
            or set(ids["split"]) != {"train", "test"}):
        raise ValueError("Invalid or incomplete embedding IDs/features")
    return ids, values, manifest


def run_comparison(features_dir: Path, labels_path: Path, output: Path,
                   audit_path: Path = Path("artifacts/reports/label_audit.json")) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"Preserve earlier runs: {output}")
    ids, values, feature_manifest = read_image_features(features_dir)
    label_audit = json.loads(audit_path.read_text())
    expected_inputs = {"train.csv", "test.csv", "train_series.csv", "test_series.csv"}
    if set(feature_manifest["input_sha256"]) != expected_inputs:
        raise ValueError("Image manifest must identify all four source metadata files")
    for name, digest in feature_manifest["input_sha256"].items():
        if digest != label_audit["input_sha256"][name]:
            raise ValueError(f"Image and label source metadata differ: {name}")
    all_features = pd.DataFrame(values, index=pd.Index(ids[ID_COLUMN], name=ID_COLUMN))
    all_features.columns = [f"image_{i:04d}" for i in range(values.shape[1])]
    labels = read_frozen_labels(labels_path, audit_path)
    train_ids = ids.loc[ids["split"].eq("train"), ID_COLUMN]
    if set(train_ids) != set(labels.index):
        raise ValueError("Embedding training IDs and label IDs differ")
    labels = labels.reindex(train_ids)
    features = all_features.loc[train_ids]
    result = compare_supervision(features, labels)
    selected = max(RECIPES, key=lambda name: result[name]["mean_macro_auc"])
    model = fit_heads(features, labels, silver_weight=RECIPES[selected])
    test = ids.loc[ids["split"].eq("test"), [ID_COLUMN]]
    submission = predict_heads(model, all_features.loc[test[ID_COLUMN]]).reset_index()
    sample = test.copy()
    sample[TARGET_COLUMNS] = 0.5
    validate_submission(submission, sample)
    output.mkdir(parents=True)
    for recipe, row in result.items():
        row["oof"].to_csv(output / f"oof_{recipe}.csv")
        (output / f"fold_models_{recipe}.json").write_text(json.dumps(row["models"]) + "\n")
    (output / "model.json").write_text(json.dumps(model) + "\n")
    submission.to_csv(output / "submission.csv", index=False)
    shutil.copy2(Path(__file__), output / "image_model_source.py")
    shutil.copy2(audit_path, output / "label_audit.json")
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Frozen generic DINOv2 image embeddings plus masked report supervision improve observed-label AUC",
        "recipes": RECIPES, "selected_recipe": selected,
        "selection": "Prespecified highest mean fold observed-label macro AUC; no leaderboard selection",
        "cv": {name: {k: v for k, v in row.items() if k not in ("models", "oof")} for name, row in result.items()},
        "estimator": {"C": 0.1, "max_iter": 1000, "seed": SEED, "class_weight": None},
        "feature_manifest": feature_manifest,
        "feature_inputs_sha256": {name: sha256(features_dir / name) for name in ("IDs.csv", "features.npy", "manifest.json")},
        "labels_sha256": sha256(labels_path),
        "label_audit_sha256": sha256(audit_path),
        "split_sha256": sha256(labels_path.with_name("folds.csv")),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src/rsnaknee").glob("*.py"))},
        "code": {"revision": revision, "dirty": bool(status), "status": status},
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "python": platform.python_version(), "runtime_seconds": time.perf_counter() - started,
        "limitations": ["58 explicit-label validation studies; model selection consumes this validation evidence", "Patient independence remains subject to DICOM audit", "Public label generator execution provenance incomplete; unknown verdicts excluded"],
        "artifact_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/processed/report_labels.csv"))
    parser.add_argument("--label-audit", type=Path, default=Path("artifacts/reports/label_audit.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_comparison(args.features, args.labels, args.output, args.label_audit)
    print(json.dumps({"selected": result["selected_recipe"], "cv": result["cv"]}, indent=2))
