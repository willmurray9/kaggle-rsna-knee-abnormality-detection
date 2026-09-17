from __future__ import annotations

import hashlib
import json
import unicodedata
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rsnaknee.constants import ID_COLUMN, METADATA_FILES, TARGET_COLUMNS
from rsnaknee.data import read_metadata, sha256

SEED = 42
N_FOLDS = 3


def report_groups(train: pd.DataFrame) -> pd.Series:
    """Group exact reports after Unicode, whitespace and case normalization."""
    normalized = train["Report"].fillna("").map(
        lambda text: " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    )
    # Missing reports are not evidence that two studies are duplicates.
    source = normalized.where(normalized.ne(""), "missing-report:" + train[ID_COLUMN])
    return source.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())


def validate_folds(train: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    if (
        list(folds.columns) != [ID_COLUMN, "group_id", "fold"]
        or folds.isna().any().any()
        or folds[ID_COLUMN].duplicated().any()
        or set(folds[ID_COLUMN]) != set(train[ID_COLUMN])
    ):
        raise ValueError("Split must contain every training study exactly once")
    if set(folds["fold"]) != set(range(N_FOLDS)):
        raise ValueError("Split must contain exactly three validation folds")
    joined = train.merge(folds, on=ID_COLUMN, validate="one_to_one")
    if not joined["group_id"].eq(report_groups(joined)).all():
        raise ValueError("Saved report groups do not match the input reports")
    if joined.groupby("group_id")["fold"].nunique().gt(1).any():
        raise ValueError("Duplicate report groups cross validation folds")
    rows = []
    for fold in range(N_FOLDS):
        labels = joined.loc[joined["fold"].eq(fold), TARGET_COLUMNS]
        for target in TARGET_COLUMNS:
            rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "observed": int(labels[target].notna().sum()),
                    "positive": int(labels[target].eq(1).sum()),
                    "negative": int(labels[target].eq(0).sum()),
                }
            )
    support = pd.DataFrame(rows)
    undefined = support.loc[support[["positive", "negative"]].eq(0).any(axis=1)]
    if not undefined.empty:
        raise ValueError(
            "Every validation target requires both classes; unsupported: "
            + str(undefined[["fold", "target"]].to_dict("records"))
        )
    return support


def freeze_folds(train: pd.DataFrame, output: Path, hashes: dict[str, str]) -> pd.DataFrame:
    """Create once, then validate and reuse; never silently replace a saved split."""
    output.mkdir(parents=True, exist_ok=True)
    path, manifest_path = output / "folds.csv", output / "folds_manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise ValueError("Frozen split is incomplete; inspect it before continuing")
        manifest = json.loads(manifest_path.read_text())
        if manifest["input_sha256"] != hashes:
            raise ValueError("Frozen split metadata changed; review before creating a new split")
        if manifest["folds_sha256"] != sha256(path):
            raise ValueError("Frozen split hash changed; inspect it before continuing")
        folds = pd.read_csv(path)
        validate_folds(train, folds)
        return folds

    ordered = train.sort_values(ID_COLUMN).reset_index(drop=True)
    groups = report_groups(ordered)
    folds = pd.DataFrame({ID_COLUMN: ordered[ID_COLUMN], "group_id": groups, "fold": -1})
    # MCL has the fewest observed positives in the audited snapshot (9 of 58).
    # Unknown labels form their own stratum, never a negative class.
    strata = ordered["MCL"].fillna(-1).astype(int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, validation) in enumerate(splitter.split(ordered, strata, groups)):
        folds.loc[validation, "fold"] = fold
    validate_folds(train, folds)
    folds.to_csv(path, index=False)
    manifest = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "method": "StratifiedGroupKFold; sorted study IDs; MCL positive/negative/unknown strata",
        "grouping": "SHA256 of NFKC, casefolded, whitespace-normalized report; empty reports use study ID",
        "patient_independent": False,
        "limitation": "CSV contains no patient ID; image duplicates and patient overlap remain unresolved",
        "split_selection": "One prespecified seed, checked only for class support; no score or seed search",
        "input_sha256": hashes,
        "folds_sha256": sha256(path),
        "scikit_learn": version("scikit-learn"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return folds


def run_eda(config: dict) -> dict:
    tables = read_metadata(config["raw_dir"])
    train = tables["train.csv"]
    labels = train[TARGET_COLUMNS]
    labeled = labels.notna().any(axis=1)
    hashes = {name: sha256(config["raw_dir"] / name) for name in METADATA_FILES}
    processed = config["raw_dir"].parent / "processed"
    folds = freeze_folds(train, processed, hashes)
    output = config["artifacts_dir"] / "eda"
    output.mkdir(parents=True, exist_ok=True)
    support = validate_folds(train, folds)
    support.to_csv(output / "fold_support.csv", index=False)

    label_summary = pd.DataFrame(
        {
            "observed": labels.notna().sum(),
            "unknown": labels.isna().sum(),
            "positive": labels.eq(1).sum(),
            "negative": labels.eq(0).sum(),
            "observed_prevalence": labels.mean(),
        }
    )
    label_summary.to_csv(output / "label_summary.csv", index_label="target")
    positive = labels.eq(1).astype(int)
    observed = labels.notna().astype(int)
    (positive.T @ positive).to_csv(output / "positive_cooccurrence.csv", index_label="target")
    (observed.T @ observed).to_csv(output / "pairwise_observed.csv", index_label="target")

    groups = report_groups(train)
    sizes = groups.value_counts()
    duplicated = sizes[sizes.gt(1)]
    duplicate_labels = pd.concat([groups.rename("group_id"), labels], axis=1)
    conflicts = duplicate_labels.groupby("group_id")[TARGET_COLUMNS].nunique().gt(1).any(axis=1)
    group_label_status = pd.DataFrame({"group": groups, "labeled": labeled}).groupby("group")["labeled"]
    reports = train["Report"].fillna("")
    char_lengths = reports.str.len()
    series_summary = {}
    for split in ("train", "test"):
        series = tables[f"{split}_series.csv"]
        counts = series.groupby(ID_COLUMN).size()
        series_summary[split] = {
            "rows": len(series),
            "series_per_study": counts.describe().to_dict(),
            "values": {
                name: {str(key): int(value) for key, value in series[name].value_counts(dropna=False).items()}
                for name in ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]
            },
            "studies_by_plane": series.groupby("Anatomical_Plane")[ID_COLUMN].nunique().to_dict(),
            "fluid_sensitivity_equals_fat_suppression": bool(series["Fluid_Sensitive"].eq(series["Fat_Suppression"]).all()),
            "missing_cells": int(series.isna().sum().sum()),
        }
    joined = train[[ID_COLUMN]].assign(observed_labels=labeled).merge(folds, on=ID_COLUMN)
    summary = {
        "input_sha256": hashes,
        "studies": {split: len(tables[f"{split}.csv"]) for split in ("train", "test")},
        "fully_labeled": int(labels.notna().all(axis=1).sum()),
        "partially_labeled": int((labeled & ~labels.notna().all(axis=1)).sum()),
        "unlabeled": int((~labeled).sum()),
        "reports": {
            "nonempty": int(reports.str.strip().ne("").sum()),
            "character_length": char_lengths.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
            "labeled_character_length": char_lengths[labeled].describe().to_dict(),
            "unlabeled_character_length": char_lengths[~labeled].describe().to_dict(),
            "containing_cyrillic": int(reports.str.contains(r"[\u0400-\u04ff]").sum()),
            "containing_cjk": int(reports.str.contains(r"[\u4e00-\u9fff]").sum()),
            "containing_non_ascii": int(reports.str.contains(r"[^\x00-\x7f]").sum()),
            "script_note": "Character scripts are clues, not identified languages; no report text is exported",
            "duplicate_groups": len(duplicated),
            "studies_in_duplicate_groups": int(duplicated.sum()),
            "largest_duplicate_group": int(sizes.max()),
            "duplicate_groups_with_labeled_and_unlabeled_studies": int((group_label_status.nunique().gt(1)).sum()),
            "groups_with_conflicting_observed_labels": int(conflicts.sum()),
        },
        "series": series_summary,
        "overlap": {
            "train_test_study_ids": len(set(train[ID_COLUMN]) & set(tables["test.csv"][ID_COLUMN])),
            "train_test_series_ids": len(set(tables["train_series.csv"]["SeriesInstanceUID"]) & set(tables["test_series.csv"]["SeriesInstanceUID"])),
            "patient_id_available": False,
            "image_duplicates_checked": False,
        },
        "split": {
            "seed": SEED,
            "folds_sha256": sha256(processed / "folds.csv"),
            "studies_per_fold": joined.groupby("fold").size().to_dict(),
            "labeled_per_fold": joined.groupby("fold")["observed_labels"].sum().to_dict(),
            "all_targets_have_both_classes_per_fold": True,
            "patient_independent": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    from rsnaknee.config import load_config

    result = run_eda(load_config())
    print(json.dumps({"studies": result["studies"], "split": result["split"]}, indent=2))
