import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.image_model import fit_heads, predict_heads, compare_supervision


def examples():
    ids = pd.Index([f"study-{i}" for i in range(24)], name=ID_COLUMN)
    features = pd.DataFrame({"a": np.tile([-2., 2.], 12), "b": np.arange(24) / 24}, index=ids)
    labels = pd.DataFrame(index=ids)
    labels["fold"] = np.repeat([0, 1, 2], 8)
    labels["group_id"] = ids
    for target in TARGET_COLUMNS:
        labels[target] = np.tile([0., 1.], 12)
        labels[target + "__observed"] = labels[target].where(np.arange(24) % 4 < 2)
        labels[target + "__mask"] = True
        labels[target + "__verdict"] = np.where(labels[target].eq(1), "YES", "NO")
    return features, labels


def test_gold_only_ignores_all_silver_labels_and_features():
    features, labels = examples()
    model = fit_heads(features, labels, silver_weight=0)
    silver = labels[TARGET_COLUMNS[0] + "__observed"].isna()
    changed_features = features.copy()
    changed_features.loc[silver] = 1e9
    changed_labels = labels.copy()
    changed_labels.loc[silver, TARGET_COLUMNS] = 1 - labels.loc[silver, TARGET_COLUMNS]
    assert fit_heads(changed_features, changed_labels, silver_weight=0) == model


def test_unknown_target_is_excluded_and_explicit_label_overrides_derived():
    features, labels = examples()
    labels.loc["study-2", "ACL"] = np.nan
    # The observed label remains authoritative even if a joined target is corrupted.
    labels.loc["study-0", "ACL"] = 1
    model = fit_heads(features, labels, silver_weight=0.25)
    assert model["counts"]["ACL"] == {"observed": 12, "derived": 11}
    assert model["counts"]["MCL"] == {"observed": 12, "derived": 12}
    expected = labels.copy()
    expected.loc["study-0", "ACL"] = 0
    assert fit_heads(features, expected, silver_weight=0.25) == model
    probabilities = predict_heads(model, features.iloc[::-1])
    assert probabilities.index.equals(features.iloc[::-1].index)
    assert list(probabilities.columns) == TARGET_COLUMNS
    assert np.isfinite(probabilities).all().all()
    assert probabilities.ge(0).all().all() and probabilities.le(1).all().all()


def test_validation_models_exclude_entire_validation_fold():
    features, labels = examples()
    result = compare_supervision(features, labels)
    changed = features.copy()
    changed.loc[labels["fold"].eq(0)] += 10000
    repeated = compare_supervision(changed, labels)
    for recipe in result:
        assert result[recipe]["models"]["0"] == repeated[recipe]["models"]["0"]
        assert result[recipe]["oof"].index.equals(labels.index[labels["ACL__observed"].notna()])


def test_cross_fold_duplicate_group_is_rejected():
    features, labels = examples()
    labels.loc[["study-0", "study-8"], "group_id"] = "duplicate"
    with pytest.raises(ValueError, match="group"):
        compare_supervision(features, labels)


def test_masked_unknown_placeholder_is_not_a_training_negative():
    features, labels = examples()
    labels.loc["study-2", ["ACL", "ACL__mask", "ACL__verdict"]] = [0., False, "UNK"]
    model = fit_heads(features, labels, silver_weight=1)
    assert model["counts"]["ACL"]["derived"] == 11


def test_edited_label_table_or_fold_assignments_are_rejected(tmp_path):
    import json
    from rsnaknee.data import sha256
    from rsnaknee.image_model import read_frozen_labels

    _, labels = examples()
    path = tmp_path / "report_labels.csv"
    folds = tmp_path / "folds.csv"
    audit = tmp_path / "label_audit.json"
    labels.to_csv(path)
    labels[["group_id", "fold"]].to_csv(folds)
    record = {"label_table_sha256": sha256(path), "folds_sha256": sha256(folds)}
    audit.write_text(json.dumps(record))
    read_frozen_labels(path, audit)
    changed = labels.copy()
    changed.loc["study-0", "fold"] = 1
    changed.to_csv(path)
    with pytest.raises(ValueError, match="hash"):
        read_frozen_labels(path, audit)
    record["label_table_sha256"] = sha256(path)
    audit.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="assignments"):
        read_frozen_labels(path, audit)


def test_partial_or_changed_image_cache_is_rejected(tmp_path):
    import json
    from rsnaknee.data import sha256
    from rsnaknee.image_model import read_image_features

    pd.DataFrame({ID_COLUMN: ["a", "b"], "split": ["train", "test"]}).to_csv(tmp_path / "IDs.csv", index=False)
    np.save(tmp_path / "features.npy", np.ones((2, 4), dtype=np.float32))
    manifest = {"status": "running", "completed_studies": 1, "feature_shape": [2, 4],
                "encoder_fit_on_competition_data": False,
                "artifact_sha256": {name: sha256(tmp_path / name) for name in ("IDs.csv", "features.npy")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        read_image_features(tmp_path)
    manifest.update(status="complete", completed_studies=2)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    read_image_features(tmp_path)
    np.save(tmp_path / "features.npy", np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="hash"):
        read_image_features(tmp_path)


@pytest.mark.parametrize("source_hashes", [{}, {"train.csv": "abc"}])
def test_missing_source_metadata_hashes_are_rejected(tmp_path, monkeypatch, source_hashes):
    import json
    import rsnaknee.image_model as module

    monkeypatch.setattr(module, "read_image_features", lambda _: (None, None, {"input_sha256": source_hashes}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"input_sha256": source_hashes}))
    with pytest.raises(ValueError, match="four source metadata"):
        module.run_comparison(tmp_path, tmp_path / "labels.csv", tmp_path / "output", audit)


def test_pca_collapsed_head_matches_explicit_projection_predictions():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from rsnaknee.image_model import SEED

    features, labels = examples()
    rng = np.random.default_rng(8)
    features['c'] = rng.normal(size=len(features))
    features['d'] = rng.normal(size=len(features))
    labels.loc['study-2', ['ACL', 'ACL__mask', 'ACL__verdict']] = [0., False, 'UNK']
    model = fit_heads(features, labels, silver_weight=.25, pca_components=2)
    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    pca = PCA(n_components=2, whiten=False, svd_solver='randomized', random_state=SEED).fit(scaled)
    projected = pca.transform(scaled)
    known = labels['ACL__observed'].notna() | labels['ACL__mask']
    truth = labels['ACL__observed'].combine_first(labels['ACL'])
    weights = np.where(labels.loc[known, 'ACL__observed'].notna(), 1., .25)
    explicit = LogisticRegression(C=.1, max_iter=1000, random_state=SEED).fit(
        projected[known], truth[known], sample_weight=weights)
    predictions = predict_heads(model, features)['ACL']
    np.testing.assert_allclose(predictions, explicit.predict_proba(projected)[:, 1], atol=1e-12, rtol=1e-12)
    assert np.asarray(model['coefficients']).shape == (12, 4)
    assert model['pca']['components'] == 2
    assert model['pca']['whiten'] is False
    np.testing.assert_allclose(model['pca']['explained_variance_ratio'], pca.explained_variance_ratio_)


def test_pca_validation_models_ignore_heldout_features():
    features, labels = examples()
    result = compare_supervision(features, labels, pca_components=1)
    changed = features.copy()
    changed.loc[labels['fold'].eq(0)] += 10000
    repeated = compare_supervision(changed, labels, pca_components=1)
    for recipe in result:
        assert result[recipe]['models']['0'] == repeated[recipe]['models']['0']


def test_default_heads_match_explicit_non_pca_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from rsnaknee.image_model import SEED

    features, labels = examples()
    model = fit_heads(features, labels, silver_weight=.25)
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features)
    gold = labels['ACL__observed']
    truth = gold.combine_first(labels['ACL'])
    explicit = LogisticRegression(C=.1, max_iter=1000, random_state=SEED).fit(
        x, truth, sample_weight=np.where(gold.notna(), 1., .25))
    np.testing.assert_array_equal(model['coefficients'][0], explicit.coef_[0])
    assert model['intercepts'][0] == explicit.intercept_[0]
    assert model['mean'] == scaler.mean_.tolist()
    assert model['scale'] == scaler.scale_.tolist()
    assert 'pca' not in model
