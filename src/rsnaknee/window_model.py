"""Fixed six-epoch diagnosis attention over frozen generic DINOv2 windows."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.baseline import _score
from rsnaknee.data import sha256
from rsnaknee.image_model import read_frozen_labels
from rsnaknee.submission import validate_submission

SEED = 20260916
EPOCHS = 6
BATCH_SIZE = 8
SILVER_WEIGHT = 0.25
GENERIC_WEIGHT_SHA256 = "1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050"


def build_supervision(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Gold overrides audited binary silver; unknown cells have safe zero targets."""
    targets = np.zeros((len(labels), len(TARGET_COLUMNS)), dtype=np.float32)
    weights = np.zeros_like(targets)
    for column, target in enumerate(TARGET_COLUMNS):
        gold = labels[target + "__observed"]
        mask = labels[target + "__mask"]
        if not pd.api.types.is_bool_dtype(mask.dtype) or mask.isna().any():
            raise ValueError("Supervision masks must be nonmissing boolean values")
        silver = labels[target].where(mask & labels[target + "__verdict"].isin(["YES", "NO"]))
        truth = gold.combine_first(silver)
        if not (truth.isna() | truth.isin([0, 1])).all():
            raise ValueError("Observed and derived targets must be binary or unknown")
        targets[:, column] = truth.fillna(0).to_numpy(dtype=np.float32)
        weights[:, column] = np.where(gold.notna(), 1., np.where(truth.notna(), SILVER_WEIGHT, 0.))
    return targets, weights


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean over batch × 12; do not renormalize silver weights within a batch."""
    return (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).mean()


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(30, 128) * 0.02)
        self.query = nn.Parameter(torch.randn(12, 128) * 0.02)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(128, 12)

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or tuple(features.shape[1:]) != (3, 10, 768):
            raise ValueError("Expected features [study, 3 planes, 10 windows, 768]")
        if presence.shape != features.shape[:2] or not torch.all((presence == 0) | (presence == 1)):
            raise ValueError("Expected binary plane presence [study, 3]")
        if not torch.all(presence.sum(1) > 0):
            raise ValueError("Every study needs a usable plane")
        mask = presence.bool().unsqueeze(-1).expand(-1, -1, 10).reshape(-1, 30)
        hidden = self.proj(features.float().reshape(-1, 30, 768)) + self.slot_emb
        attention = torch.einsum("bsh,th->bts", hidden, self.query) / 128 ** 0.5
        attention = attention.masked_fill(~mask.unsqueeze(1), float("-inf")).softmax(-1)
        context = self.drop(torch.einsum("bts,bsh->bth", attention, hidden))
        return (context * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


def _check_arrays(features: np.ndarray, presence: np.ndarray) -> None:
    if features.ndim != 4 or features.shape[1:] != (3, 10, 768) or not np.isfinite(features).all():
        raise ValueError("Invalid window features")
    if presence.shape != features.shape[:2] or not np.isin(presence, [0, 1]).all():
        raise ValueError("Invalid plane presence")
    if not np.all(presence.sum(1) > 0):
        raise ValueError("Every study needs a usable plane")


def fit_head(features: np.ndarray, presence: np.ndarray, targets: np.ndarray,
             weights: np.ndarray) -> tuple[AttentionHead, list[float]]:
    """Fit the final fixed epoch on CPU, using only supplied training studies."""
    _check_arrays(features, presence)
    if (targets.shape != (len(features), 12) or weights.shape != targets.shape
            or not np.isin(targets, [0, 1]).all() or not np.isin(weights, [0, .25, 1]).all()):
        raise ValueError("Invalid training targets or weights")
    usable = weights.sum(1) > 0
    if not usable.any():
        raise ValueError("No supervised training studies")
    for column, target in enumerate(TARGET_COLUMNS):
        if np.unique(targets[weights[:, column] > 0, column]).size != 2:
            raise ValueError(f"Training target lacks both classes: {target}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    random = np.random.default_rng(SEED)
    model = AttentionHead().cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    history = []
    for _ in range(EPOCHS):
        permutation = random.permutation(np.flatnonzero(usable))
        total = 0.
        for start in range(0, len(permutation), BATCH_SIZE):
            selected = permutation[start:start + BATCH_SIZE]
            x = torch.from_numpy(np.asarray(features[selected], dtype=np.float32))
            mask = torch.from_numpy(np.asarray(presence[selected]))
            y = torch.from_numpy(np.asarray(targets[selected], dtype=np.float32))
            w = torch.from_numpy(np.asarray(weights[selected], dtype=np.float32))
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(x, mask), y, w)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        history.append(total / len(permutation))
    return model.eval(), history


def train_fold(features: np.ndarray, presence: np.ndarray, labels: pd.DataFrame,
               heldout_fold: int | None) -> tuple[AttentionHead, dict]:
    if len(features) != len(labels) or labels.index.has_duplicates:
        raise ValueError("Features and unique label study IDs must align")
    if labels[["fold", "group_id"]].isna().any().any():
        raise ValueError("Missing fold/group assignments")
    if labels.groupby("group_id")["fold"].nunique().gt(1).any():
        raise ValueError("A report group crosses validation folds")
    if heldout_fold is not None and heldout_fold not in labels["fold"].unique():
        raise ValueError("Unknown held-out fold")
    training = np.ones(len(labels), dtype=bool) if heldout_fold is None else labels["fold"].ne(heldout_fold).to_numpy()
    # Exclude every held-out study before resolving any gold or report supervision.
    selected_labels = labels.loc[training]
    targets, weights = build_supervision(selected_labels)
    model, history = fit_head(features[training], presence[training], targets, weights)
    usable = weights.sum(1) > 0
    return model, {
        "heldout_fold": heldout_fold, "training_ids": selected_labels.index[usable].tolist(),
        "excluded_fold_studies": int((~training).sum()), "epoch_training_loss": history,
        "target_counts": {target: {"observed": int((weights[:, i] == 1).sum()),
                                   "derived": int((weights[:, i] == .25).sum())}
                          for i, target in enumerate(TARGET_COLUMNS)},
    }


@torch.no_grad()
def predict_windows(model: AttentionHead, features: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Return 12 probabilities in input order; no training metadata is required."""
    _check_arrays(features, presence)
    model.eval()
    predictions = []
    for start in range(0, len(features), BATCH_SIZE):
        x = torch.from_numpy(np.array(features[start:start + BATCH_SIZE], dtype=np.float32, copy=True))
        mask = torch.from_numpy(np.array(presence[start:start + BATCH_SIZE], copy=True))
        predictions.append(model(x, mask).sigmoid().numpy())
    values = np.concatenate(predictions) if predictions else np.empty((0, 12), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite window predictions")
    return values


def load_head(path: Path) -> AttentionHead:
    model = AttentionHead().cpu()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def read_window_cache(cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Need a complete window cache")
    if (manifest.get("encoder_frozen") is not True
            or manifest.get("encoder_fit_on_competition_data") is not False
            or manifest.get("labels_or_reports_read") is not False):
        raise ValueError("Need frozen generic features extracted without labels or reports")
    checkpoint = manifest.get("checkpoint", {})
    if (checkpoint.get("model_type") != "dinov2" or checkpoint.get("hidden_size") != 384
            or checkpoint.get("files_sha256", {}).get("pytorch_model.bin") != GENERIC_WEIGHT_SHA256):
        raise ValueError("Window checkpoint differs from audited generic DINOv2-small weights")
    if (manifest.get("recipe") != "central12-neighbor3-mean10-v1"
            or manifest.get("planes") != ["Sagittal", "Coronal", "Axial"]
            or manifest.get("window_order") != [list(range(i, i + 3)) for i in range(10)]
            or manifest.get("embedding_layout") != "CLS[384] then mean-patch[384]"):
        raise ValueError("Window recipe or embedding layout differs")
    for name in ("IDs.csv", "window_features.npy", "presence.npy"):
        if sha256(cache_dir / name) != manifest.get("artifact_sha256", {}).get(name):
            raise ValueError(f"Window cache hash differs: {name}")
    ids = pd.read_csv(cache_dir / "IDs.csv", dtype=str)
    if (set(ids.columns) != {ID_COLUMN, "split"} or ids.empty or ids[ID_COLUMN].isna().any()
            or ids[ID_COLUMN].str.strip().eq("").any() or ids[ID_COLUMN].duplicated().any()
            or set(ids["split"]) != {"train", "test"}):
        raise ValueError("Invalid or duplicate window cache IDs")
    features = np.load(cache_dir / "window_features.npy", allow_pickle=False, mmap_mode="r")
    presence = np.load(cache_dir / "presence.npy", allow_pickle=False, mmap_mode="r")
    if (features.shape != (len(ids), 3, 10, 768) or list(features.shape) != manifest.get("window_feature_shape")
            or features.dtype != np.float16 or manifest.get("window_feature_dtype") != "float16"
            or presence.dtype != np.uint8 or list(presence.shape) != manifest.get("presence_shape")
            or manifest.get("completed_studies") != len(ids)):
        raise ValueError("Invalid or incomplete window cache dimensions/dtype")
    _check_arrays(features, presence)
    return ids, features, presence, manifest


def run_comparison(cache_dir: Path, labels_path: Path, output: Path,
                   audit_path: Path = Path("artifacts/reports/label_audit_image_v1.json")) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"Preserve earlier runs: {output}")
    ids, all_features, all_presence, manifest = read_window_cache(cache_dir)
    audit = json.loads(audit_path.read_text())
    expected_inputs = {"train.csv", "test.csv", "train_series.csv", "test_series.csv"}
    if set(manifest.get("input_sha256", {})) != expected_inputs:
        raise ValueError("Window manifest must identify all four source metadata files")
    for name, digest in manifest["input_sha256"].items():
        if digest != audit["input_sha256"].get(name):
            raise ValueError(f"Window and label source metadata differ: {name}")
    labels = read_frozen_labels(labels_path, audit_path)
    training = ids["split"].eq("train").to_numpy()
    train_ids = ids.loc[training, ID_COLUMN]
    if set(train_ids) != set(labels.index):
        raise ValueError("Window training IDs and label IDs differ")
    labels = labels.reindex(train_ids)
    if set(labels["fold"]) != {0, 1, 2}:
        raise ValueError("Expected the saved three-fold split")
    features, presence = all_features[training], all_presence[training]
    gold = labels[[target + "__observed" for target in TARGET_COLUMNS]].copy()
    gold.columns = TARGET_COLUMNS
    observed = gold.notna().any(axis=1)
    if gold.loc[observed].isna().any().any():
        raise ValueError("This experiment expects complete explicit validation labels")
    output.mkdir(parents=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({"status": "running"}) + "\n")
    oof = pd.DataFrame(np.nan, index=labels.index[observed], columns=TARGET_COLUMNS)
    folds = []
    for fold in range(3):
        model, record = train_fold(features, presence, labels, fold)
        validation = (labels["fold"].eq(fold) & observed).to_numpy()
        prediction = pd.DataFrame(predict_windows(model, features[validation], presence[validation]),
                                  index=labels.index[validation], columns=TARGET_COLUMNS)
        oof.loc[prediction.index] = prediction
        torch.save(model.state_dict(), output / f"fold_{fold}.pt")
        training_ids_path = output / f"fold_{fold}_training_ids.csv"
        pd.DataFrame({ID_COLUMN: record.pop("training_ids")}).to_csv(training_ids_path, index=False)
        folds.append({"fold": fold, **record, **_score(gold.loc[validation], prediction),
                      "training_ids_sha256": sha256(training_ids_path)})
        print(f"Finished fold {fold}: macro AUC {folds[-1]['macro_auc']:.6f}", flush=True)
    if oof.isna().any().any():
        raise ValueError("Incomplete observed-label OOF predictions")
    oof.to_csv(output / "oof.csv")
    model, final_record = train_fold(features, presence, labels, None)
    torch.save(model.state_dict(), output / "model.pt")
    final_ids_path = output / "final_training_ids.csv"
    pd.DataFrame({ID_COLUMN: final_record.pop("training_ids")}).to_csv(final_ids_path, index=False)
    final_record["training_ids_sha256"] = sha256(final_ids_path)
    test = ids.loc[~training, [ID_COLUMN]].reset_index(drop=True)
    submission = test.copy()
    submission[TARGET_COLUMNS] = predict_windows(model, all_features[~training], all_presence[~training])
    sample = test.copy()
    sample[TARGET_COLUMNS] = .5
    validate_submission(submission, sample)
    submission.to_csv(output / "submission.csv", index=False)
    shutil.copy2(Path(__file__), output / "window_model_source.py")
    shutil.copy2(audit_path, output / "label_audit.json")
    shutil.copy2(cache_dir / "manifest.json", output / "feature_manifest.json")
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    macro = [row["macro_auc"] for row in folds]
    summary = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Diagnosis-specific attention over 30 frozen generic image windows improves fixed-fold ranking",
        "selection": "Fixed final epoch; no validation checkpoint selection or leaderboard tuning",
        "recipe": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED, "hidden": 128,
                   "dropout": .2, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": .02,
                   "silver_weight": SILVER_WEIGHT, "loss": "weighted BCE mean over batch x 12",
                   "class_weight": None, "target_balancing": None, "device": "cpu", "threads": 1,
                   "deterministic_algorithms": True, "targets": TARGET_COLUMNS},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "folds": folds, "final_fit": final_record, "gold_validation_studies": len(oof),
        "mean_macro_auc": float(np.mean(macro)), "std_macro_auc": float(np.std(macro, ddof=1)),
        "mean_per_target_auc": {target: float(np.mean([row["per_label_auc"][target] for row in folds]))
                                for target in TARGET_COLUMNS},
        "feature_manifest": manifest,
        "feature_inputs_sha256": {name: sha256(cache_dir / name) for name in
                                  ("IDs.csv", "window_features.npy", "presence.npy", "manifest.json")},
        "labels_sha256": sha256(labels_path), "label_audit_sha256": sha256(audit_path),
        "split_sha256": sha256(labels_path.with_name("folds.csv")),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in sorted((root / "src/rsnaknee").glob("*.py"))},
        "code": {"revision": revision, "dirty": bool(status), "status": status},
        "packages": {name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn")},
        "python": platform.python_version(), "runtime_seconds": time.perf_counter() - started,
        "limitations": ["Repeated exploratory use of 58 gold studies", "Patient independence unresolved",
                        "Public report-label generator independence from gold studies unknown"],
        "artifact_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path != summary_path},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/processed/image-v1/report_labels.csv"))
    parser.add_argument("--label-audit", type=Path, default=Path("artifacts/reports/label_audit_image_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_comparison(args.features, args.labels, args.output, args.label_audit)
    print(json.dumps({"mean_macro_auc": result["mean_macro_auc"], "runtime_seconds": result["runtime_seconds"]}, indent=2))
