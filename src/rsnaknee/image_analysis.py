"""Fixed paired OOF comparisons; no model fitting or recipe selection."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256

RECIPES = ('observed_only', 'silver_full', 'silver_quarter')
COMPARISONS = (('silver_full', 'observed_only'), ('silver_quarter', 'observed_only'),
               ('observed_only', 'metadata'), ('silver_full', 'metadata'),
               ('silver_quarter', 'metadata'))
SEED = 20260916


def analyze_predictions(truth, folds, predictions, *, replicates=1000):
    """Paired study resampling within each saved fold, with fixed predictions."""
    if replicates < 1:
        raise ValueError('Need at least one bootstrap replicate')
    if (truth.empty or truth.index.has_duplicates or truth.index.isna().any()
            or list(truth.columns) != TARGET_COLUMNS or not truth.isin([0, 1]).all().all()):
        raise ValueError('Need complete observed binary labels with unique study IDs')
    if not truth.index.equals(folds.index) or folds[['fold', 'group_id']].isna().any().any():
        raise ValueError('Truth and nonmissing frozen assignments must align')
    if folds.groupby('group_id')['fold'].nunique().gt(1).any():
        raise ValueError('A group crosses validation folds')
    if folds.group_id.duplicated().any():
        raise ValueError('Study bootstrap requires distinct observed-label report groups')
    if set(predictions) != {*RECIPES, 'metadata'}:
        raise ValueError('Need the three fixed recipes and metadata reference')
    arrays = {}
    for name, frame in predictions.items():
        if not frame.index.equals(truth.index) or list(frame.columns) != TARGET_COLUMNS:
            raise ValueError('Prediction IDs and target order must match observed truth')
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Predictions must be finite probabilities in [0, 1]')
        arrays[name] = values
    y = truth.to_numpy(dtype=int)
    fold_ids = sorted(folds.fold.unique())
    if len(fold_ids) < 2:
        raise ValueError('Need at least two saved validation folds')
    groups = [np.flatnonzero(folds.fold.to_numpy() == fold) for fold in fold_ids]

    def valid(indices):
        return all(np.all(y[index].min(axis=0) != y[index].max(axis=0)) for index in indices)

    def scores(indices):
        return {name: np.mean([roc_auc_score(y[index], values[index], average=None)
                               for index in indices], axis=0)
                for name, values in arrays.items()}

    if not valid(groups):
        raise ValueError('Every target requires both observed classes in every fold')
    point = scores(groups)
    models = {}
    for name, values in arrays.items():
        models[name] = {
            'mean_macro_auc': float(point[name].mean()),
            'mean_per_target_auc': dict(zip(TARGET_COLUMNS, point[name].tolist())),
            'mean_per_target_brier': dict(zip(TARGET_COLUMNS, np.mean([
                ((values[index] - y[index]) ** 2).mean(axis=0) for index in groups], axis=0).tolist())),
            'folds': [{'fold': int(fold), 'studies': len(index),
                       'positive_counts': dict(zip(TARGET_COLUMNS, y[index].sum(axis=0).tolist())),
                       'macro_auc': float(roc_auc_score(y[index], values[index], average='macro'))}
                      for fold, index in zip(fold_ids, groups)],
        }
    rng = np.random.default_rng(SEED)
    differences = {f'{a}_minus_{b}': [] for a, b in COMPARISONS}
    accepted = 0
    for _ in range(replicates):
        sampled = [rng.choice(index, size=len(index), replace=True) for index in groups]
        if not valid(sampled):
            continue
        current = scores(sampled)
        accepted += 1
        for a, b in COMPARISONS:
            differences[f'{a}_minus_{b}'].append(current[a] - current[b])
    comparisons = {}
    for a, b in COMPARISONS:
        name = f'{a}_minus_{b}'
        delta = point[a] - point[b]
        draws = np.asarray(differences[name])
        intervals = np.quantile(draws, [0.025, 0.975], axis=0) if accepted else None
        comparisons[name] = {
            'macro_auc_difference': float(delta.mean()),
            'bootstrap_macro_95_interval': np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist() if accepted else None,
            'per_target_auc_difference': dict(zip(TARGET_COLUMNS, delta.tolist())),
            'bootstrap_per_target_95_intervals': {
                target: intervals[:, j].tolist() if accepted else None for j, target in enumerate(TARGET_COLUMNS)
            },
            'per_target_brier_difference': {
                target: models[a]['mean_per_target_brier'][target] - models[b]['mean_per_target_brier'][target]
                for target in TARGET_COLUMNS
            },
        }
    error_rows = []
    for name in RECIPES:
        for j, target in enumerate(TARGET_COLUMNS):
            squared = (arrays[name][:, j] - y[:, j]) ** 2
            for i in np.argsort(-squared, kind='stable')[:5]:
                error_rows.append({ID_COLUMN: truth.index[i], 'fold': int(folds.fold.iloc[i]),
                    'recipe': name, 'target': target, 'observed': int(y[i, j]),
                    'prediction': float(arrays[name][i, j]), 'squared_error': float(squared[i]),
                    'observed_only_prediction': float(arrays['observed_only'][i, j]),
                    'metadata_prediction': float(arrays['metadata'][i, j]),
                    'brier_difference_vs_observed_only': float(squared[i] - (arrays['observed_only'][i, j] - y[i, j]) ** 2),
                    'brier_difference_vs_metadata': float(squared[i] - (arrays['metadata'][i, j] - y[i, j]) ** 2)})
    return {
        'estimand': 'Unweighted mean of per-fold observed-label AUC; never pooled OOF AUC',
        'models': models, 'comparisons': comparisons,
        'bootstrap': {'attempted': replicates, 'accepted': accepted,
                      'excluded_missing_class': replicates - accepted, 'seed': SEED,
                      'unit': 'study, independently within each saved fold; paired across models',
                      'interval': '2.5/97.5 percentile; conditional on all targets retaining both classes in every fold'},
        'limitations': [
            'Fixed OOF predictions: no model refitting or training-data uncertainty is represented.',
            'Intervals condition on class-valid replicates; exclusions can be substantial for rare findings.',
            'Distinct observed report groups permit study resampling but do not establish independent patients.',
            'Recipe/checkpoint selection elsewhere consumes validation evidence; these are descriptive comparisons, not independent confirmation.',
            'Per-target intervals are unadjusted for multiple comparisons; no hypothesis tests or hyperparameter selection.',
        ],
    }, pd.DataFrame(error_rows)


def _read_unique(path):
    frame = pd.read_csv(path, dtype={ID_COLUMN: str})
    if ID_COLUMN not in frame or frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f'Invalid study IDs in {path.name}')
    return frame.set_index(ID_COLUMN)


def run_analysis(experiment, labels_path, folds_path, metadata_path, output, *, replicates=1000):
    """Check provenance, write a new analysis directory, never change the experiment."""
    experiment, labels_path, folds_path, metadata_path, output = map(
        Path, (experiment, labels_path, folds_path, metadata_path, output))
    if output.exists():
        raise FileExistsError(f'Preserve earlier analysis: {output}')
    summary_path = experiment / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['labels_sha256'] != sha256(labels_path) or summary['split_sha256'] != sha256(folds_path):
        raise ValueError('Experiment labels or frozen split hash differs')
    labels, folds = _read_unique(labels_path), _read_unique(folds_path)
    if set(labels.index) != set(folds.index):
        raise ValueError('Labels and frozen split have different IDs')
    folds = folds.reindex(labels.index)
    if not labels[['group_id', 'fold']].equals(folds[['group_id', 'fold']]):
        raise ValueError('Labels differ from frozen fold assignments')
    if folds[['fold', 'group_id']].isna().any().any() or folds.groupby('group_id').fold.nunique().gt(1).any():
        raise ValueError('Missing assignment or group crosses validation folds')
    gold = labels[[t + '__observed' for t in TARGET_COLUMNS]].copy()
    gold.columns = TARGET_COLUMNS
    present = gold.notna().any(axis=1)
    gold = gold.loc[present]
    if gold.isna().any().any():
        raise ValueError('Partial observed targets need a separately prespecified analysis')
    gold = gold.sort_index()
    predictions = {}
    inputs = {'experiment_summary': summary_path, 'labels': labels_path,
              'frozen_folds': folds_path, 'metadata_oof': metadata_path}
    for recipe in RECIPES:
        path = experiment / f'oof_{recipe}.csv'
        if sha256(path) != summary['artifact_sha256'][path.name]:
            raise ValueError(f'OOF artifact hash differs: {recipe}')
        frame = _read_unique(path)
        if set(frame.index) != set(gold.index) or list(frame.columns) != TARGET_COLUMNS:
            raise ValueError(f'OOF ID/target coverage differs: {recipe}')
        predictions[recipe] = frame.reindex(gold.index)
        inputs[recipe + '_oof'] = path
    metadata_summary_path = metadata_path.with_name('summary.json')
    metadata_summary = json.loads(metadata_summary_path.read_text())
    if metadata_summary['split_sha256'] != sha256(folds_path):
        raise ValueError('Metadata frozen split hash differs')
    if metadata_summary['artifact_sha256']['oof.csv'] != sha256(metadata_path):
        raise ValueError('Metadata OOF hash differs')
    metadata_source = metadata_path.with_name('baseline_source.py')
    source_hash = sha256(metadata_source)
    if (source_hash != metadata_summary['artifact_sha256']['baseline_source.py']
            or source_hash != metadata_summary['source_sha256']['src/rsnaknee/baseline.py']):
        raise ValueError('Metadata source hash differs')
    inputs.update(metadata_summary=metadata_summary_path, metadata_source=metadata_source)
    metadata = _read_unique(metadata_path)
    if set(metadata.index) != set(gold.index):
        raise ValueError('Metadata OOF does not cover exactly the observed studies')
    metadata = metadata.reindex(gold.index)
    if not metadata[['group_id', 'fold']].equals(folds.loc[gold.index, ['group_id', 'fold']]):
        raise ValueError('Metadata OOF uses a different split')
    observed = metadata[['observed_' + t for t in TARGET_COLUMNS]].copy()
    observed.columns = TARGET_COLUMNS
    if not np.array_equal(observed.to_numpy(), gold.to_numpy()):
        raise ValueError('Metadata observed labels differ')
    predictions['metadata'] = metadata[['learned_' + t for t in TARGET_COLUMNS]].copy()
    predictions['metadata'].columns = TARGET_COLUMNS
    analysis, errors = analyze_predictions(gold, folds.loc[gold.index], predictions, replicates=replicates)
    for recipe in RECIPES:
        if not np.isclose(analysis['models'][recipe]['mean_macro_auc'], summary['cv'][recipe]['mean_macro_auc'], atol=1e-10, rtol=0):
            raise ValueError(f'Recomputed OOF score differs from experiment summary: {recipe}')
    analysis.update({
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'observed_studies': len(gold),
        'input_sha256': {name: sha256(path) for name, path in inputs.items()},
        'source_sha256': {'image_analysis.py': sha256(Path(__file__))},
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn')},
        'error_table': 'Five largest squared probability errors per recipe and target; positive Brier delta means worse',
    })
    output.mkdir(parents=True)
    errors.to_csv(output / 'largest_errors.csv', index=False)
    analysis['largest_errors_sha256'] = sha256(output / 'largest_errors.csv')
    (output / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
    return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--labels', default=Path('data/processed/report_labels.csv'), type=Path)
    parser.add_argument('--folds', default=Path('data/processed/folds.csv'), type=Path)
    parser.add_argument('--metadata', default=Path('artifacts/experiments/metadata-v1/oof.csv'), type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run_analysis(args.experiment, args.labels, args.folds, args.metadata, args.output)
    print(json.dumps({'observed_studies': result['observed_studies'], 'bootstrap': result['bootstrap']}, indent=2))
