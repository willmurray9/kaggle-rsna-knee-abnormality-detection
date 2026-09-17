from __future__ import annotations

import numpy as np
import pandas as pd

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.eda import report_groups


def _ids(frame: pd.DataFrame, name: str) -> None:
    if ID_COLUMN not in frame or frame.empty:
        raise ValueError(f'{name} requires nonempty study IDs')
    ids = frame[ID_COLUMN]
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f'{name} study IDs must be strings')
    if ids.str.strip().eq('').any() or not ids.eq(ids.str.strip()).all():
        raise ValueError(f'{name} contains blank or padded study IDs')
    if ids.duplicated().any():
        raise ValueError(f'{name} contains duplicate study IDs')


def build_label_table(
    train: pd.DataFrame, derived: pd.DataFrame, folds: pd.DataFrame, *, source: str
) -> pd.DataFrame:
    """Keep observed labels, report verdicts and masks separate on the saved split.

    Source must identify the audited input/version (prefer a file SHA-256). Scores
    and confidences are preserved as metadata, not treated as calibrated targets.
    This function does not establish source permission or extraction accuracy.
    """
    for frame, name in [(train, 'train'), (derived, 'derived'), (folds, 'folds')]:
        _ids(frame, name)
    if not isinstance(source, str) or not source.strip():
        raise ValueError('A label source/version is required')
    if not set(derived[ID_COLUMN]).issubset(set(train[ID_COLUMN])):
        raise ValueError('Derived labels contain IDs absent from training studies')
    if set(folds[ID_COLUMN]) != set(train[ID_COLUMN]):
        raise ValueError('Saved folds must cover every training study exactly once')
    if not {'group_id', 'fold'}.issubset(folds.columns):
        raise ValueError('Saved folds require group_id and fold')
    if folds[['group_id', 'fold']].isna().any().any():
        raise ValueError('Saved folds contain missing groups or folds')
    fold_values = pd.to_numeric(folds['fold'], errors='raise')
    if not np.isfinite(fold_values).all() or (fold_values < 0).any() or (fold_values % 1 != 0).any():
        raise ValueError('Saved fold numbers must be nonnegative integers')
    if folds.groupby('group_id')['fold'].nunique().gt(1).any():
        raise ValueError('Saved report groups cross validation folds')
    if 'Report' not in train or not set(TARGET_COLUMNS).issubset(train.columns):
        raise ValueError('Training input requires reports and observed target columns')
    expected_groups = pd.Series(report_groups(train).to_numpy(), index=train[ID_COLUMN])
    if not folds['group_id'].eq(folds[ID_COLUMN].map(expected_groups)).all():
        raise ValueError('Saved report groups do not match input reports')
    observed = train[TARGET_COLUMNS]
    if not (observed.isna() | observed.isin([0, 1])).all().all():
        raise ValueError('Observed labels must be binary; unknown labels stay missing')
    required = [c for target in TARGET_COLUMNS for c in [target, target + '__conf', target + '__verdict']]
    if not set(required).issubset(derived.columns):
        raise ValueError('Derived labels require a score, confidence and verdict per target')
    for target in TARGET_COLUMNS:
        for column in [target, target + '__conf']:
            values = pd.to_numeric(derived[column], errors='raise')
            if not np.isfinite(values).all() or not values.between(0, 1).all():
                raise ValueError(f'Derived {column} must contain finite values in [0, 1]')
        if not derived[target + '__verdict'].isin(['YES', 'NO', 'UNK']).all():
            raise ValueError(f'Derived {target} verdict must be YES, NO or UNK')

    ordered = train.set_index(ID_COLUMN)
    public = derived.set_index(ID_COLUMN).reindex(ordered.index)
    split = folds.set_index(ID_COLUMN).reindex(ordered.index)
    columns = {ID_COLUMN: ordered.index.to_numpy(),
               'group_id': split['group_id'].to_numpy(),
               'fold': split['fold'].to_numpy(dtype=int)}
    for target in TARGET_COLUMNS:
        gold = ordered[target].astype(float)
        verdict = public[target + '__verdict']
        confidence = public[target + '__conf'].astype(float)
        known = verdict.isin(['YES', 'NO']) & confidence.gt(0)
        silver = verdict.map({'YES': 1., 'NO': 0.}).where(known)
        value = gold.combine_first(silver)
        provenance = pd.Series('unknown', index=ordered.index)
        provenance.loc[public[target].notna()] = source
        provenance.loc[gold.notna()] = 'observed'
        for name, values in {
            target: value, target + '__observed': gold,
            target + '__derived': public[target].astype(float),
            target + '__verdict': verdict, target + '__confidence': confidence,
            target + '__mask': value.notna(), target + '__source': provenance,
        }.items():
            columns[name] = values.to_numpy()
    return pd.DataFrame(columns)


def training_labels(labels: pd.DataFrame, *, heldout_fold: int) -> pd.DataFrame:
    """Exclude every held-out study, then discard rows with no usable supervision."""
    if heldout_fold not in set(labels['fold']):
        raise ValueError('Unknown heldout fold')
    known = labels[[target + '__mask' for target in TARGET_COLUMNS]].any(axis=1)
    return labels.loc[labels['fold'].ne(heldout_fold) & known].copy()


def audit_label_source(config: dict) -> dict:
    """Read the pinned local inputs and emit aggregate diagnostics, never report text."""
    import json
    from datetime import datetime, timezone

    from rsnaknee.constants import METADATA_FILES
    from rsnaknee.data import read_metadata, sha256
    from rsnaknee.eda import freeze_folds

    raw = config['raw_dir']
    source_dir = raw.parent / 'external' / 'public-labels'
    processed = raw.parent / 'processed'
    if not (processed / 'folds.csv').is_file() or not (processed / 'folds_manifest.json').is_file():
        raise FileNotFoundError('Create and review the saved split before label ingestion')
    files = ['dataset-metadata.json', 'api_labeler.py', 'report_labels_v2.csv']
    source_hashes = {name: sha256(source_dir / name) for name in files}
    input_hashes = {name: sha256(raw / name) for name in METADATA_FILES}
    train = read_metadata(raw)['train.csv']
    folds = freeze_folds(train, processed, input_hashes)
    derived = pd.read_csv(source_dir / 'report_labels_v2.csv', dtype={ID_COLUMN: str})
    source = 'pilkwang/rsna-knee-llm-labels@sha256:' + source_hashes['report_labels_v2.csv']
    labels = build_label_table(train, derived, folds, source=source)
    label_path = processed / 'report_labels.csv'
    labels.to_csv(label_path, index=False)
    targets = {}
    for target in TARGET_COLUMNS:
        gold = labels[target + '__observed']
        verdict = labels[target + '__verdict']
        binary = verdict.map({'YES': 1., 'NO': 0.})
        usable = binary.notna() & labels[target + '__confidence'].gt(0)
        compared = gold.notna() & usable
        targets[target] = {
            'observed': int(gold.notna().sum()),
            'source_verdicts': {str(k): int(v) for k, v in verdict.value_counts().items()},
            'added_supervised_targets': int((gold.isna() & usable).sum()),
            'masked_targets': int((~labels[target + '__mask']).sum()),
            'agreement_denominator': int(compared.sum()),
            'agreement_matches': int((gold[compared] == binary[compared]).sum()),
        }
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': source, 'source_file_sha256': source_hashes,
        'input_sha256': input_hashes, 'folds_sha256': sha256(processed / 'folds.csv'),
        'policy': 'Observed first; otherwise YES=1/NO=0 at confidence>0; UNK always masked',
        'policy_frozen_before_gold_diagnostics': True,
        'permission_status': 'Public reuse supported by rules 2.6 and pinned host clarification 733965; specific generator compliance unverified',
        'rules_source': 'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules',
        'host_clarification': 'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965',
        'label_table_sha256': sha256(label_path),
        'provenance_status': 'Incomplete finding definitions, model run manifest and evidence',
        'agreement_is_independent_validation': False,
        'raw_reports_exported': False,
        'training_studies': len(train), 'public_label_studies': len(derived),
        'missing_public_studies': len(train) - len(derived), 'targets': targets,
        'training_studies_by_heldout_fold': {
            str(fold): len(training_labels(labels, heldout_fold=fold))
            for fold in sorted(labels['fold'].unique())
        },
    }
    output = config['artifacts_dir'] / 'reports'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'label_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import json

    from rsnaknee.config import load_config

    result = audit_label_source(load_config())
    print(json.dumps({key: result[key] for key in [
        'training_studies', 'public_label_studies', 'missing_public_studies',
        'permission_status', 'training_studies_by_heldout_fold',
    ]}, indent=2))
