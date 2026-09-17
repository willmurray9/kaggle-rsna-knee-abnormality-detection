import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.eda import report_groups
from rsnaknee.labels import build_label_table, training_labels


@pytest.fixture
def inputs():
    train = pd.DataFrame({ID_COLUMN: ['gold', 'silver', 'unknown', 'missing'],
                          'Report': ['Synthetic gold', 'Synthetic silver', 'Synthetic unknown', 'Synthetic missing'],
                          **{t: [0., np.nan, np.nan, np.nan] for t in TARGET_COLUMNS}})
    derived = pd.DataFrame({ID_COLUMN: ['unknown', 'silver', 'gold']})
    for target in TARGET_COLUMNS:
        derived[target] = [.28, .08, .94]
        derived[target + '__conf'] = [.05, .85, .95]
        derived[target + '__verdict'] = ['UNK', 'NO', 'YES']
    derived.loc[1, 'ACL'] = .68
    derived.loc[1, 'ACL__verdict'] = 'YES'
    derived.loc[1, 'ACL__conf'] = .95
    folds = pd.DataFrame({ID_COLUMN: ['missing', 'unknown', 'gold', 'silver'],
                          'fold': [2, 1, 0, 1], 'group_id': ['m', 'u', 'g', 's']})
    folds['group_id'] = folds[ID_COLUMN].map(dict(zip(train[ID_COLUMN], report_groups(train))))
    return train, derived, folds


def test_exact_id_join_preserves_gold_and_unknowns(inputs):
    train, derived, folds = inputs
    original = train.copy(deep=True)
    labels = build_label_table(train, derived, folds, source='public@sha256').set_index(ID_COLUMN)
    pd.testing.assert_frame_equal(train, original)
    assert labels.loc['gold', TARGET_COLUMNS].eq(0).all()
    assert labels.loc['gold', 'ACL__observed'] == 0
    assert labels.loc['gold', 'ACL__derived'] == .94
    assert labels.loc['gold', 'ACL__source'] == 'observed'
    assert labels.loc['silver', 'ACL'] == 1
    assert labels.loc['silver', 'MCL'] == 0
    assert labels.loc['silver', 'ACL__confidence'] == .95
    assert labels.loc['silver', 'ACL__source'] == 'public@sha256'
    for uid in ['unknown', 'missing']:
        assert labels.loc[uid, TARGET_COLUMNS].isna().all()
        assert not labels.loc[uid, [t + '__mask' for t in TARGET_COLUMNS]].any()
    assert labels.loc['unknown', 'ACL__derived'] == .28
    assert labels.loc['unknown', 'ACL__verdict'] == 'UNK'
    assert labels.loc['silver', 'fold'] == 1


def test_heldout_studies_and_reports_never_enter_training(inputs):
    train, derived, folds = inputs
    labels = build_label_table(train, derived, folds, source='public')
    selected = training_labels(labels, heldout_fold=1)
    assert selected[ID_COLUMN].tolist() == ['gold']
    assert 'Report' not in labels
    assert not set(selected[ID_COLUMN]) & set(folds.loc[folds.fold == 1, ID_COLUMN])
    with pytest.raises(ValueError, match='heldout'):
        training_labels(labels, heldout_fold=9)


@pytest.mark.parametrize('column,value', [('ACL', -.1), ('ACL', 1.1), ('ACL', np.inf),
    ('ACL', np.nan), ('ACL__conf', -.1), ('ACL__conf', 1.1), ('ACL__conf', np.nan),
    ('ACL__verdict', 'MAYBE'), ('ACL__verdict', None)])
def test_invalid_derived_values_are_rejected(inputs, column, value):
    train, derived, folds = inputs
    derived.loc[0, column] = value
    with pytest.raises(ValueError):
        build_label_table(train, derived, folds, source='public')


@pytest.mark.parametrize('table_index', [0, 1, 2])
@pytest.mark.parametrize('bad_id', ['', ' ', None, ' gold', 'not-a-training-study'])
def test_invalid_or_unmatched_ids_are_rejected(inputs, table_index, bad_id):
    tables = list(inputs)
    tables[table_index].loc[0, ID_COLUMN] = bad_id
    with pytest.raises(ValueError):
        build_label_table(*tables, source='public')


@pytest.mark.parametrize('table_index', [0, 1, 2])
def test_duplicate_ids_rejected(inputs, table_index):
    tables = list(inputs)
    tables[table_index] = pd.concat([tables[table_index], tables[table_index].iloc[[0]]])
    with pytest.raises(ValueError, match='duplicate'):
        build_label_table(*tables, source='public')


def test_cross_fold_groups_rejected(inputs):
    train, derived, folds = inputs
    folds.loc[0, 'group_id'] = folds.loc[2, 'group_id']
    with pytest.raises(ValueError, match='group'):
        build_label_table(train, derived, folds, source='public')


def test_missing_fold_and_nonbinary_gold_rejected(inputs):
    train, derived, folds = inputs
    with pytest.raises(ValueError, match='fold'):
        build_label_table(train, derived, folds.iloc[1:], source='public')
    train.loc[0, 'ACL'] = .5
    with pytest.raises(ValueError, match='Observed'):
        build_label_table(train, derived, folds, source='public')


def test_zero_confidence_verdict_is_masked(inputs):
    train, derived, folds = inputs
    derived.loc[1, 'ACL__conf'] = 0
    labels = build_label_table(train, derived, folds, source='public').set_index(ID_COLUMN)
    assert np.isnan(labels.loc['silver', 'ACL'])
    assert not labels.loc['silver', 'ACL__mask']


def test_changed_report_invalidates_saved_group(inputs):
    train, derived, folds = inputs
    train.loc[0, 'Report'] = 'Different synthetic report'
    with pytest.raises(ValueError, match='report groups'):
        build_label_table(train, derived, folds, source='public')
