import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.window_model import (
    AttentionHead, build_supervision, fit_head, load_head, masked_bce,
    predict_windows, train_fold,
)


def examples():
    ids = pd.Index([f"study-{i}" for i in range(18)], name=ID_COLUMN)
    labels = pd.DataFrame({"fold": np.repeat([0, 1, 2], 6), "group_id": ids}, index=ids)
    for target in TARGET_COLUMNS:
        labels[target] = np.tile([0., 1.], 9)
        labels[target + "__observed"] = labels[target].where(np.arange(18) % 3 != 2)
        labels[target + "__mask"] = True
        labels[target + "__verdict"] = np.where(labels[target].eq(1), "YES", "NO")
    rng = np.random.default_rng(31)
    features = rng.normal(size=(18, 3, 10, 768)).astype(np.float16)
    features[:, :, :, :32] += np.tile([-2., 2.], 9)[:, None, None, None]
    return features, np.ones((18, 3), dtype=np.uint8), labels


def test_gold_precedence_unknown_masks_and_loss_gradient():
    _, _, labels = examples()
    labels.loc["study-0", "ACL"] = 1
    labels.loc["study-2", ["ACL", "ACL__mask", "ACL__verdict"]] = [np.nan, False, "UNK"]
    targets, weights = build_supervision(labels)
    assert targets[0, 0] == 0 and weights[0, 0] == 1
    assert targets[2, 0] == 0 and weights[2, 0] == 0
    assert weights[2, 1] == .25
    assert np.isfinite(targets).all()
    logits = torch.zeros((1, 12), requires_grad=True)
    loss = masked_bce(logits, torch.zeros_like(logits), torch.tensor([[1., .25] + [0.] * 10]))
    loss.backward()
    np.testing.assert_allclose(logits.grad.numpy(), [[.5 / 12, .125 / 12] + [0.] * 10])


def test_attention_ignores_masked_planes_and_rejects_all_missing():
    torch.manual_seed(2)
    model = AttentionHead().eval()
    features, presence, _ = examples()
    presence[:, 1] = 0
    expected = predict_windows(model, features, presence)
    features[:, 1] = 1000
    np.testing.assert_array_equal(predict_windows(model, features, presence), expected)
    with pytest.raises(ValueError, match="usable plane"):
        predict_windows(model, features, np.zeros_like(presence))


def test_real_training_updates_weights_and_roundtrips_safely(tmp_path):
    features, presence, labels = examples()
    targets, weights = build_supervision(labels)
    torch.manual_seed(20260916)
    initial = AttentionHead()
    model, history = fit_head(features, presence, targets, weights)
    assert len(history) == 6 and all(np.isfinite(history))
    assert any(not torch.equal(value, model.state_dict()[name]) for name, value in initial.state_dict().items())
    expected = predict_windows(model, features, presence)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    actual = predict_windows(load_head(path), features[::-1], presence[::-1])
    np.testing.assert_array_equal(actual[::-1], expected)
    assert expected.shape == (18, 12)
    assert np.isfinite(expected).all() and np.all((expected >= 0) & (expected <= 1))


def test_fold_training_excludes_heldout_gold_and_silver_images_and_labels():
    features, presence, labels = examples()
    model, record = train_fold(features, presence, labels, 0)
    changed_features = features.copy()
    changed_features[:6] *= -3
    changed_labels = labels.copy()
    changed_labels.loc[labels["fold"].eq(0), TARGET_COLUMNS] = 1 - labels.loc[labels["fold"].eq(0), TARGET_COLUMNS]
    changed_labels.loc[labels["fold"].eq(0), [t + "__observed" for t in TARGET_COLUMNS]] = 1
    repeated, _ = train_fold(changed_features, presence, changed_labels, 0)
    assert record["training_ids"] == [f"study-{i}" for i in range(6, 18)]
    assert all(torch.equal(value, repeated.state_dict()[name]) for name, value in model.state_dict().items())
    labels.loc["study-6", "group_id"] = "study-0"
    with pytest.raises(ValueError, match="group"):
        train_fold(features, presence, labels, 0)


def write_inputs(tmp_path):
    from rsnaknee.data import sha256

    features, presence, labels = examples()
    cache = tmp_path / "cache"
    cache.mkdir()
    values = np.concatenate([features, features[:3]])
    flags = np.concatenate([presence, presence[:3]])
    pd.DataFrame({ID_COLUMN: labels.index.tolist() + ["test-a", "test-b", "test-c"],
                  "split": ["train"] * 18 + ["test"] * 3}).to_csv(cache / "IDs.csv", index=False)
    np.save(cache / "window_features.npy", values)
    np.save(cache / "presence.npy", flags)
    metadata = {name: name + "-hash" for name in ("train.csv", "train_series.csv", "test.csv", "test_series.csv")}
    manifest = {
        "status": "complete", "recipe": "central12-neighbor3-mean10-v1", "completed_studies": 21,
        "encoder_frozen": True, "encoder_fit_on_competition_data": False, "labels_or_reports_read": False,
        "checkpoint": {"model_type": "dinov2", "hidden_size": 384, "files_sha256": {
            "pytorch_model.bin": "1051e25b2ed69ddad24f3c41e7b6eed6e7f7d012103ea227e47eb82e87dc2050"}},
        "window_feature_shape": [21, 3, 10, 768], "window_feature_dtype": "float16",
        "presence_shape": [21, 3], "planes": ["Sagittal", "Coronal", "Axial"],
        "window_order": [list(range(i, i + 3)) for i in range(10)],
        "embedding_layout": "CLS[384] then mean-patch[384]", "input_sha256": metadata,
        "artifact_sha256": {name: sha256(cache / name) for name in ("IDs.csv", "window_features.npy", "presence.npy")},
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))
    labels.to_csv(tmp_path / "report_labels.csv")
    labels[["fold", "group_id"]].to_csv(tmp_path / "folds.csv")
    audit = {"input_sha256": metadata, "label_table_sha256": sha256(tmp_path / "report_labels.csv"),
             "folds_sha256": sha256(tmp_path / "folds.csv")}
    (tmp_path / "audit.json").write_text(json.dumps(audit))
    return cache, manifest


@pytest.mark.parametrize("change,match", [
    ({"status": "running"}, "complete"),
    ({"encoder_fit_on_competition_data": True}, "generic"),
    ({"planes": ["Axial", "Coronal", "Sagittal"]}, "layout"),
    ({"window_order": [[0, 1, 2]] * 10}, "layout"),
    ({"checkpoint": {"model_type": "dinov2", "hidden_size": 384, "files_sha256": {"pytorch_model.bin": "external"}}}, "checkpoint"),
])
def test_wrong_cache_provenance_and_layout_are_rejected(tmp_path, change, match):
    from rsnaknee.window_model import read_window_cache

    cache, manifest = write_inputs(tmp_path)
    read_window_cache(cache)
    manifest.update(change)
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        read_window_cache(cache)


def test_window_cache_hash_and_duplicate_ids_are_checked(tmp_path):
    from rsnaknee.data import sha256
    from rsnaknee.window_model import read_window_cache

    cache, manifest = write_inputs(tmp_path)
    ids = pd.read_csv(cache / "IDs.csv")
    ids.loc[1, ID_COLUMN] = ids.loc[0, ID_COLUMN]
    ids.to_csv(cache / "IDs.csv", index=False)
    with pytest.raises(ValueError, match="hash"):
        read_window_cache(cache)
    manifest["artifact_sha256"]["IDs.csv"] = sha256(cache / "IDs.csv")
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="IDs"):
        read_window_cache(cache)


def test_complete_synthetic_experiment_saves_oof_final_and_provenance(tmp_path):
    from rsnaknee.data import sha256
    from rsnaknee.window_model import read_window_cache, run_comparison

    cache, _ = write_inputs(tmp_path)
    output = tmp_path / "run"
    summary = run_comparison(cache, tmp_path / "report_labels.csv", output, tmp_path / "audit.json")
    assert len(summary["folds"]) == 3
    assert summary["status"] == "complete"
    assert summary["split_sha256"] == sha256(tmp_path / "folds.csv")
    oof = pd.read_csv(output / "oof.csv")
    assert len(oof) == 12 and not oof[TARGET_COLUMNS].isna().any().any()
    prediction = pd.read_csv(output / "submission.csv")
    assert prediction[ID_COLUMN].tolist() == ["test-a", "test-b", "test-c"]
    _, values, presence, _ = read_window_cache(cache)
    np.testing.assert_allclose(prediction[TARGET_COLUMNS], predict_windows(load_head(output / "model.pt"), values[-3:], presence[-3:]))
    for fold in range(3):
        assert (output / f"fold_{fold}.pt").exists()
        assert len(summary["folds"][fold]["epoch_training_loss"]) == 6
    with pytest.raises(FileExistsError):
        run_comparison(cache, tmp_path / "report_labels.csv", output, tmp_path / "audit.json")


def test_feature_and_label_metadata_must_match_before_training(tmp_path):
    from rsnaknee.window_model import run_comparison

    cache, manifest = write_inputs(tmp_path)
    manifest["input_sha256"]["train.csv"] = "different"
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source metadata"):
        run_comparison(cache, tmp_path / "report_labels.csv", tmp_path / "run", tmp_path / "audit.json")
    assert not (tmp_path / "run").exists()
