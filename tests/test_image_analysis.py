import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import TARGET_COLUMNS
from rsnaknee.image_analysis import analyze_predictions


def fixture():
    ids = pd.Index([f's{i}' for i in range(12)], name='StudyInstanceUID')
    truth = pd.DataFrame(np.tile([0, 1] * 6, (12, 1)).T, index=ids, columns=TARGET_COLUMNS)
    folds = pd.DataFrame({'fold': [0] * 6 + [1] * 6, 'group_id': ids}, index=ids)
    perfect = truth * 0.8 + 0.1
    predictions = {'observed_only': perfect, 'silver_full': perfect.copy(),
                   'silver_quarter': perfect.copy(), 'metadata': 1 - perfect}
    return truth, folds, predictions


def test_paired_bootstrap_is_deterministic_and_matches_within_fold_auc():
    truth, folds, predictions = fixture()
    result, errors = analyze_predictions(truth, folds, predictions, replicates=25)
    repeated, _ = analyze_predictions(truth, folds, predictions, replicates=25)
    assert result == repeated
    assert result['models']['observed_only']['mean_macro_auc'] == 1
    assert result['models']['metadata']['mean_macro_auc'] == 0
    assert result['comparisons']['silver_full_minus_observed_only']['macro_auc_difference'] == 0
    assert result['comparisons']['observed_only_minus_metadata']['bootstrap_macro_95_interval'] == [1, 1]
    assert result['bootstrap']['accepted'] + result['bootstrap']['excluded_missing_class'] == 25
    assert set(errors.target) == set(TARGET_COLUMNS)
    assert 'brier_difference_vs_metadata' in errors.columns


def test_bootstrap_refuses_group_leakage_and_repeated_gold_groups():
    truth, folds, predictions = fixture()
    folds.iloc[6, folds.columns.get_loc('group_id')] = 's0'
    with pytest.raises(ValueError, match='crosses'):
        analyze_predictions(truth, folds, predictions, replicates=25)
    folds.iloc[6, folds.columns.get_loc('group_id')] = 's6'
    folds.iloc[1, folds.columns.get_loc('group_id')] = 's0'
    with pytest.raises(ValueError, match='distinct'):
        analyze_predictions(truth, folds, predictions, replicates=25)


def test_auc_is_mean_within_fold_not_pooled():
    truth, folds, predictions = fixture()
    for name, values in predictions.items():
        if name != 'metadata':
            values.loc[folds.fold == 0] = truth.loc[folds.fold == 0] * 0.1 + 0.1
            values.loc[folds.fold == 1] = truth.loc[folds.fold == 1] * 0.1 + 0.8
    result, _ = analyze_predictions(truth, folds, predictions, replicates=25)
    assert result['models']['observed_only']['mean_macro_auc'] == 1
    predictions['silver_full'].iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='probabilities'):
        analyze_predictions(truth, folds, predictions, replicates=25)


def test_analysis_writes_new_artifacts_and_checks_provenance(tmp_path):
    import json
    from rsnaknee.data import sha256
    from rsnaknee.image_analysis import run_analysis
    truth, folds, predictions = fixture()
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    labels = truth.rename(columns={t: t + '__observed' for t in TARGET_COLUMNS}).join(folds)
    labels_path = tmp_path / 'labels.csv'
    folds_path = tmp_path / 'folds.csv'
    metadata_dir = tmp_path / 'metadata'
    metadata_dir.mkdir()
    metadata_path = metadata_dir / 'oof.csv'
    labels.to_csv(labels_path)
    folds.to_csv(folds_path)
    metadata = folds.join(truth.add_prefix('observed_')).join(predictions['metadata'].add_prefix('learned_'))
    metadata.to_csv(metadata_path)
    metadata_source = metadata_dir / 'baseline_source.py'
    metadata_source.write_text('# Fixed metadata training source snapshot\n')
    metadata_summary = {
        'split_sha256': sha256(folds_path),
        'source_sha256': {'src/rsnaknee/baseline.py': sha256(metadata_source)},
        'artifact_sha256': {'oof.csv': sha256(metadata_path), 'baseline_source.py': sha256(metadata_source)},
    }
    metadata_summary_path = metadata_dir / 'summary.json'
    metadata_summary_path.write_text(json.dumps(metadata_summary))
    artifact_hashes = {}
    for recipe in ('observed_only', 'silver_full', 'silver_quarter'):
        path = experiment / f'oof_{recipe}.csv'
        predictions[recipe].to_csv(path)
        artifact_hashes[path.name] = sha256(path)
    summary = {'labels_sha256': sha256(labels_path), 'split_sha256': sha256(folds_path),
               'artifact_sha256': artifact_hashes,
               'cv': {name: {'mean_macro_auc': 1.0} for name in ('observed_only', 'silver_full', 'silver_quarter')}}
    (experiment / 'summary.json').write_text(json.dumps(summary))
    before = {p.name: p.read_bytes() for p in experiment.iterdir()}
    output = tmp_path / 'analysis'
    result = run_analysis(experiment, labels_path, folds_path, metadata_path, output, replicates=25)
    assert result['observed_studies'] == 12
    assert result['input_sha256']['metadata_summary'] == sha256(metadata_summary_path)
    assert result['input_sha256']['metadata_source'] == sha256(metadata_source)
    assert json.loads((output / 'analysis.json').read_text())['bootstrap']['attempted'] == 25
    assert {p.name: p.read_bytes() for p in experiment.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_analysis(experiment, labels_path, folds_path, metadata_path, output, replicates=25)
    metadata_bytes = metadata_path.read_bytes()
    metadata.loc['s0', 'learned_ACL'] = 0.3
    metadata.to_csv(metadata_path)
    with pytest.raises(ValueError, match='Metadata OOF hash differs'):
        run_analysis(experiment, labels_path, folds_path, metadata_path, tmp_path / 'edited_metadata', replicates=25)
    assert not (tmp_path / 'edited_metadata').exists()
    metadata_path.write_bytes(metadata_bytes)
    source_bytes = metadata_source.read_bytes()
    metadata_source.write_text('# Edited source\n')
    with pytest.raises(ValueError, match='Metadata source hash differs'):
        run_analysis(experiment, labels_path, folds_path, metadata_path, tmp_path / 'edited_source', replicates=25)
    metadata_source.write_bytes(source_bytes)
    metadata_summary['split_sha256'] = 'wrong'
    metadata_summary_path.write_text(json.dumps(metadata_summary))
    with pytest.raises(ValueError, match='Metadata frozen split hash differs'):
        run_analysis(experiment, labels_path, folds_path, metadata_path, tmp_path / 'edited_split', replicates=25)
    metadata_summary['split_sha256'] = sha256(folds_path)
    metadata_summary_path.write_text(json.dumps(metadata_summary))
    path = experiment / 'oof_observed_only.csv'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash differs'):
        run_analysis(experiment, labels_path, folds_path, metadata_path, tmp_path / 'second', replicates=25)
    assert not (tmp_path / 'second').exists()
