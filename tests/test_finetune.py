from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from torch import nn

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee import finetune
from rsnaknee.window_model import AttentionHead, masked_bce


class SmallEncoder(nn.Module):
    """Offline boundary double: real autograd through early/late encoder blocks."""
    def __init__(self, blocks=4):
        super().__init__()
        self.embeddings = nn.Linear(3, 384)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(384, 384) for _ in range(blocks)])
        self.layernorm = nn.LayerNorm(384)
        self.dropout = nn.Dropout(.9)

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values.mean((-1, -2)))
        for layer in self.encoder.layer:
            x = self.dropout(torch.nn.functional.gelu(layer(x)))
        x = self.layernorm(x)
        return SimpleNamespace(last_hidden_state=torch.stack([x, x * .5], 1))


def examples():
    ids = pd.Index([f'study-{i}' for i in range(9)], name=ID_COLUMN)
    labels = pd.DataFrame({'fold': np.repeat([0, 1, 2], 3), 'group_id': ids}, index=ids)
    for target in TARGET_COLUMNS:
        labels[target] = np.arange(9) % 2
        labels[target + '__observed'] = labels[target].where(np.arange(9) % 3 != 2)
        labels[target + '__mask'] = True
        labels[target + '__verdict'] = np.where(labels[target].eq(1), 'YES', 'NO')
    pixels = np.random.default_rng(19).integers(0, 256, (9, 3, 12, 4, 4), dtype=np.uint8)
    return pixels, np.ones((9, 3), dtype=np.uint8), labels


def test_sampling_shared_by_study_independent_of_order_and_epoch_varies():
    table = finetune.window_indices(['a', 'b', 'c'])
    assert table.shape == (6, 3, 3)
    assert np.all((table >= 0) & (table < 10))
    np.testing.assert_array_equal(table[:, [2, 0]], finetune.window_indices(['c', 'a']))
    assert np.any(table[0] != table[1])


def test_slot_mask_preserves_head_identity_and_ignores_unselected_slots():
    head = AttentionHead().eval()
    features = torch.randn(2, 3, 10, 768)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[:, [2, 17, 25]] = True
    before = {name: id(value) for name, value in head.named_parameters()}
    expected = finetune.slot_logits(head, features, mask)
    features.reshape(2, 30, 768)[~mask] = 1e6
    torch.testing.assert_close(finetune.slot_logits(head, features, mask), expected)
    assert before == {name: id(value) for name, value in head.named_parameters()}
    with pytest.raises(ValueError):
        finetune.slot_logits(head, features, torch.zeros_like(mask))


@pytest.mark.parametrize('arm', ['frozen', 'late_blocks'])
def test_real_gradient_updates_only_authorized_parameters_and_unknowns_zero(arm):
    torch.manual_seed(4)
    model = finetune.MRIModel(SmallEncoder(), arm).train()
    assert not model.encoder.training and model.head.training
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    pixels, presence, labels = examples()
    logits = model(pixels[:2], presence[:2], finetune.window_indices(labels.index) [0, :2])
    weights = torch.zeros_like(logits)
    weights[:, 0] = 1
    weights[:, 1] = .25
    logits.retain_grad()
    loss = masked_bce(logits, torch.zeros_like(logits), weights)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 2:]) == 0
    optimizer = finetune.make_optimizer(model)
    optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(before[name], p)}
    assert any(name.startswith('head.') for name in changed)
    assert not any(name.startswith(('encoder.embeddings.', 'encoder.encoder.layer.0.', 'encoder.encoder.layer.1.')) for name in changed)
    expected = {'encoder.encoder.layer.2.', 'encoder.encoder.layer.3.', 'encoder.layernorm.'}
    for prefix in expected:
        assert any(name.startswith(prefix) for name in changed) == (arm == 'late_blocks')


@pytest.mark.parametrize('arm,blocks', [('late_blocks', 4), ('deep_blocks', 8)])
def test_heldout_changes_cannot_change_fitted_weights_and_safe_load(tmp_path, arm, blocks):
    pixels, presence, labels = examples()
    torch.manual_seed(32)
    encoder_state = SmallEncoder(blocks).state_dict()
    def factory(arm):
        encoder = SmallEncoder(blocks)
        encoder.load_state_dict(encoder_state)
        torch.manual_seed(finetune.SEED)
        return finetune.MRIModel(encoder, arm)
    table = finetune.window_indices(labels.index)
    model, record = finetune.train_fold(factory, arm, pixels, presence, labels, np.arange(9), table, 0)
    changed_pixels, changed_labels = pixels.copy(), labels.copy()
    changed_pixels[:3] = 255
    changed_labels.loc[labels.fold.eq(0), TARGET_COLUMNS] = -99
    changed_labels.loc[labels.fold.eq(0), [t + '__observed' for t in TARGET_COLUMNS]] = -99
    repeated, _ = finetune.train_fold(factory, arm, changed_pixels, presence, changed_labels, np.arange(9), table, 0)
    assert record['training_ids'] == [f'study-{i}' for i in range(3, 9)]
    assert record['excluded_fold_studies'] == 3
    assert len(record['epoch_training_loss']) == 6
    assert record['encoder_adaptation']['trainable_block_indices'] == list(range(2, blocks))
    assert record['encoder_adaptation']['trainable_blocks'] == blocks - 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    loaded = factory(arm)
    finetune.load_model_state(loaded, path, provenance=record['encoder_adaptation'])
    wrong_depth = dict(record['encoder_adaptation'], trainable_blocks=99)
    with pytest.raises(ValueError, match='adaptation provenance'):
        finetune.load_model_state(loaded, path, provenance=wrong_depth)
    expected = finetune.predict_pixels(model, pixels, presence, np.arange(9))
    actual = finetune.predict_pixels(loaded, pixels, presence, np.arange(8, -1, -1))
    np.testing.assert_allclose(actual[::-1], expected, atol=1e-7)
    assert expected.shape == (9, 12) and np.isfinite(expected).all()
    assert np.all((expected >= 0) & (expected <= 1))


def test_projection_refuses_overbudget_full_job():
    assert finetune.project_runtime({'frozen': 1., 'late_blocks': 2.}, 100, 4., 58, 50.) < 27000
    assert finetune.project_runtime({'frozen': 2., 'late_blocks': 8.}, 5000, 4., 58, 50.) > 27000


def test_runtime_projection_uses_only_selected_arm_count():
    assert finetune.project_runtime({'deep_blocks': 2.}, 100, 4., 58, 50.) == 1190.


@pytest.mark.parametrize('arm,blocks', [('late_blocks', 3), ('deep_blocks', 7)])
def test_tiny_real_dinov2_autograd_freeze_boundary(arm, blocks):
    transformers = pytest.importorskip('transformers')
    encoder = transformers.Dinov2Model(transformers.Dinov2Config(
        hidden_size=384, num_hidden_layers=blocks, num_attention_heads=6,
        mlp_ratio=1, image_size=28, patch_size=14))
    model = finetune.MRIModel(encoder, arm).train()
    before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    pixels = np.random.default_rng(8).integers(0, 256, (1, 3, 12, 28, 28), dtype=np.uint8)
    loss = model(pixels, np.ones((1, 3), np.uint8), np.array([[1, 4, 8]])).square().mean()
    loss.backward()
    finetune.make_optimizer(model).step()
    changed = {name for name, value in encoder.named_parameters() if not torch.equal(value, before[name])}
    assert changed
    assert all(name.startswith(tuple(f'encoder.layer.{i}.' for i in range(1, blocks)) + ('layernorm.',))
               for name in changed)
    for index in range(1, blocks):
        assert any(name.startswith(f'encoder.layer.{index}.') for name in changed)
    assert any(name.startswith('layernorm.') for name in changed)


def test_deep_arm_requires_six_blocks_and_preserves_shared_head_initialization():
    with pytest.raises(ValueError, match='at least 6'):
        finetune.MRIModel(SmallEncoder(), 'deep_blocks')
    torch.manual_seed(71)
    late = finetune.MRIModel(SmallEncoder(8), 'late_blocks').train()
    torch.manual_seed(71)
    deep = finetune.MRIModel(SmallEncoder(8), 'deep_blocks').train()
    for name, value in late.state_dict().items():
        torch.testing.assert_close(value, deep.state_dict()[name], rtol=0, atol=0)
    assert not deep.encoder.training and deep.head.training
    assert [group['lr'] for group in finetune.make_optimizer(deep).param_groups] == [.001, .000008]


def test_all_window_inference_matches_unchanged_attention_head():
    torch.manual_seed(4)
    model = finetune.MRIModel(SmallEncoder(), 'frozen').eval()
    pixels, presence, _ = examples()
    pixels, presence = pixels[:1], presence[:1]
    vectors = []
    from rsnaknee.features import normalize_rgb
    with torch.no_grad():
        for plane in range(3):
            for window in range(10):
                x = normalize_rgb(pixels[:, plane, window:window + 3].astype(np.float32) / 255.)
                tokens = model.encoder(torch.from_numpy(x)).last_hidden_state
                vectors.append(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1))
        expected = model.head(torch.cat(vectors).reshape(1, 3, 10, 768), torch.ones(1, 3))
        actual = model(pixels, presence)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def pixel_cache(tmp_path):
    import json
    from rsnaknee.data import sha256
    cache = tmp_path / 'cache'
    cache.mkdir()
    pd.DataFrame({ID_COLUMN: ['a', 'b'], 'split': ['train', 'test']}).to_csv(cache / 'IDs.csv', index=False)
    np.save(cache / 'pixels_uint8.npy', np.zeros((2, 3, 12, 224, 224), np.uint8))
    np.save(cache / 'presence.npy', np.ones((2, 3), np.uint8))
    manifest = {'status': 'complete', 'completed_studies': 2,
                'encoder_frozen': True, 'encoder_fit_on_competition_data': False, 'labels_or_reports_read': False,
                'checkpoint': {'model_type': 'dinov2', 'hidden_size': 384,
                               'files_sha256': {'pytorch_model.bin': finetune.GENERIC_WEIGHT_SHA256}},
                'recipe': 'central12-neighbor3-mean10-v1', 'planes': ['Sagittal', 'Coronal', 'Axial'],
                'window_order': [list(range(i, i + 3)) for i in range(10)],
                'embedding_layout': 'CLS[384] then mean-patch[384]',
                'pixel_cache_shape': [2, 3, 12, 224, 224], 'pixel_cache_dtype': 'uint8',
                'pixel_cache_quantization': 'round(255*clip(preprocessed_float,0,1)); cache only; feature extraction uses original float32',
                'presence_shape': [2, 3],
                'input_sha256': {name: name + '-hash' for name in ('train.csv', 'test.csv', 'train_series.csv', 'test_series.csv')},
                'artifact_sha256': {name: sha256(cache / name) for name in ('IDs.csv', 'presence.npy', 'pixels_uint8.npy')}}
    manifest['source_sha256'] = {name: sha256(Path(finetune.__file__).with_name(name))
                                for name in ('__init__.py', 'coverage.py', 'features.py', 'imaging.py')}
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    return cache, manifest


def test_pixel_cache_is_readonly_and_rejects_corruption_and_metadata_mismatch(tmp_path):
    import json
    cache, manifest = pixel_cache(tmp_path)
    audit = {'input_sha256': dict(manifest['input_sha256'])}
    ids, pixels, presence, _ = finetune.read_pixel_cache(cache, audit)
    assert isinstance(pixels, np.memmap) and not pixels.flags.writeable
    assert pixels.shape == (2, 3, 12, 224, 224)
    audit['input_sha256']['train.csv'] = 'wrong'
    with pytest.raises(ValueError, match='metadata'):
        finetune.read_pixel_cache(cache, audit)
    audit['input_sha256'] = manifest['input_sha256']
    with (cache / 'pixels_uint8.npy').open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'\x01')
    with pytest.raises(ValueError, match='hash'):
        finetune.read_pixel_cache(cache, audit)


@pytest.mark.parametrize('arms', [('frozen', 'late_blocks'), ('late_blocks', 'deep_blocks')])
def test_probe_restores_rng_does_not_seed_real_fits_and_records_excluded_training(arms):
    pixels, presence, labels = examples()
    def factory(arm):
        return finetune.MRIModel(SmallEncoder(8), arm)
    table = finetune.window_indices(labels.index)
    before = torch.random.get_rng_state().clone()
    probe = finetune.throughput_probe(factory, pixels, presence, labels, np.arange(9), table, 0, arms=arms)
    assert torch.equal(before, torch.random.get_rng_state())
    assert probe['steps_per_arm'] == 64
    assert list(probe['seconds_per_step']) == list(arms)
    assert probe['selected_arms'] == list(arms)
    assert probe['total_training_steps_per_arm'] == 30  # 3 folds x 1 batch + final x 2, for 6 epochs.
    for arm in arms:
        assert probe['encoder_adaptation'][arm]['trainable_blocks'] == {'frozen': 0, 'late_blocks': 2, 'deep_blocks': 6}[arm]
    assert probe['probe_training_ids'] == [f'study-{i}' for i in range(3, 9)]
    assert probe['within_budget'] and probe['projected_total_seconds'] < 27000


@pytest.mark.parametrize('arms', [None, ('late_blocks', 'deep_blocks')])
def test_complete_two_arm_run_binds_schedule_ids_oof_and_checkpoints(tmp_path, monkeypatch, arms):
    import json
    import transformers
    from rsnaknee.data import sha256
    pixels, presence, labels = examples()
    indices = np.arange(60) % 9
    labels = labels.iloc[indices].copy()
    labels.index = pd.Index([f'study-{i}' for i in range(60)], name=ID_COLUMN)
    labels['group_id'] = labels.index
    labels['fold'] = np.arange(60) % 3
    for target in TARGET_COLUMNS:
        labels[target + '__observed'] = np.arange(60) % 2
        labels.loc[labels.index[-2:], target + '__observed'] = np.nan
    labels.to_csv(tmp_path / 'report_labels.csv')
    labels[['fold', 'group_id']].to_csv(tmp_path / 'folds.csv')
    audit = {'label_table_sha256': sha256(tmp_path / 'report_labels.csv'),
             'folds_sha256': sha256(tmp_path / 'folds.csv')}
    (tmp_path / 'audit.json').write_text(json.dumps(audit))
    all_pixels = np.concatenate([pixels[indices], pixels[:2]])
    all_presence = np.ones((62, 3), np.uint8)
    ids = pd.DataFrame({ID_COLUMN: labels.index.tolist() + ['test-a', 'test-b'],
                        'split': ['train'] * 60 + ['test'] * 2})
    manifest = {'checkpoint': {'files_sha256': {'pytorch_model.bin': 'audited'}},
                'input_sha256': {}, 'artifact_sha256': {'pixels_uint8.npy': 'pixels-digest'}}
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(finetune, 'read_pixel_cache', lambda *args: (ids, all_pixels, all_presence, manifest))
    monkeypatch.setattr(finetune, 'checkpoint_provenance', lambda _: manifest['checkpoint'])
    torch.manual_seed(92)
    encoder_state = SmallEncoder(8).state_dict()
    def load_encoder(*args, **kwargs):
        encoder = SmallEncoder(8)
        encoder.load_state_dict(encoder_state)
        return encoder
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', load_encoder)
    output = tmp_path / 'run'
    options = {} if arms is None else {'arms': arms}
    expected_arms = ('frozen', 'late_blocks') if arms is None else arms
    summary = finetune.run_comparison(cache, tmp_path, tmp_path / 'report_labels.csv', output,
                                      tmp_path / 'audit.json', device='cpu', **options)
    assert summary['status'] == 'complete'
    assert list(summary['arms']) == list(expected_arms)
    assert summary['selected_arms'] == list(expected_arms)
    assert summary['probe']['selected_arms'] == list(expected_arms)
    assert summary['recipe']['trainable_blocks_by_arm'] == {arm: {'frozen': 0, 'late_blocks': 2, 'deep_blocks': 6}[arm]
                                                          for arm in expected_arms}
    bound_ids = pd.read_csv(output / 'window_study_ids.csv')[ID_COLUMN].tolist()
    np.testing.assert_array_equal(np.load(output / 'window_indices.npy'), finetune.window_indices(bound_ids))
    for arm in expected_arms:
        oof = pd.read_csv(output / arm / 'oof.csv')
        assert oof[ID_COLUMN].tolist() == labels.index[:58].tolist()
        assert np.isfinite(oof[TARGET_COLUMNS].to_numpy()).all()
        for fold in range(3):
            train_ids = pd.read_csv(output / arm / f'fold_{fold}_training_ids.csv')[ID_COLUMN]
            assert set(train_ids).isdisjoint(set(labels.index[labels.fold.eq(fold)]))
            state = torch.load(output / arm / f'fold_{fold}.pt', weights_only=True)
            assert all(isinstance(value, torch.Tensor) for value in state.values())
        submission = pd.read_csv(output / arm / 'submission.csv')
        assert submission[ID_COLUMN].tolist() == ['test-a', 'test-b']
        assert len(summary['arms'][arm]['final_fit']['epoch_training_loss']) == 6
        for fit in summary['arms'][arm]['folds'] + [summary['arms'][arm]['final_fit']]:
            assert fit['encoder_adaptation']['trainable_blocks'] == {'frozen': 0, 'late_blocks': 2, 'deep_blocks': 6}[arm]
            assert fit['encoder_adaptation']['encoder_blocks'] == 8
        assert summary['artifact_sha256'][arm + '/model.pt'] == sha256(output / arm / 'model.pt')
    with pytest.raises(FileExistsError):
        finetune.run_comparison(cache, tmp_path, tmp_path / 'report_labels.csv', output,
                               tmp_path / 'audit.json', device='cpu', **options)
    monkeypatch.setattr(finetune, 'project_runtime', lambda *args: 30000.)
    refused = tmp_path / 'refused'
    deferred = finetune.run_comparison(cache, tmp_path, tmp_path / 'report_labels.csv', refused,
                                       tmp_path / 'audit.json', device='cpu', **options)
    assert deferred['status'] == 'deferred_budget'
    assert all(not (refused / arm).exists() for arm in expected_arms)


@pytest.mark.parametrize('arms', [(), ('late_blocks', 'late_blocks'), ('unknown',), 'deep_blocks'])
def test_invalid_arm_selection_fails_before_reading_inputs_or_creating_run(tmp_path, arms):
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='arms'):
        finetune.run_comparison(tmp_path, tmp_path, tmp_path / 'missing.csv', output, arms=arms)
    assert not output.exists()


def test_pixel_cache_rejects_preprocessing_source_drift(tmp_path):
    import json
    cache, manifest = pixel_cache(tmp_path)
    manifest['source_sha256'] = {'coverage.py': 'changed'}
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source'):
        finetune.read_pixel_cache(cache, {'input_sha256': manifest['input_sha256']})


def test_real_dinov2_sampled_logits_use_original_thirty_slot_identities():
    transformers = pytest.importorskip('transformers')
    torch.manual_seed(16)
    encoder = transformers.Dinov2Model(transformers.Dinov2Config(
        hidden_size=384, num_hidden_layers=3, num_attention_heads=6,
        mlp_ratio=1, image_size=28, patch_size=14))
    model = finetune.MRIModel(encoder, 'late_blocks').eval()
    pixels = np.random.default_rng(17).integers(0, 256, (1, 3, 12, 28, 28), dtype=np.uint8)
    windows = np.array([[2, 7, 5]])
    x = np.stack([pixels[0, plane, window:window + 3] for plane, window in enumerate([2, 7, 5])])
    mean = np.array([.485, .456, .406], np.float32)[None, :, None, None]
    std = np.array([.229, .224, .225], np.float32)[None, :, None, None]
    with torch.no_grad():
        tokens = encoder(pixel_values=torch.from_numpy((x.astype(np.float32) / 255. - mean) / std)).last_hidden_state
        features = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1)
        hidden = model.head.proj(features) + model.head.slot_emb[[2, 17, 25]]
        scores = model.head.query @ hidden.T / 128 ** .5
        context = scores.softmax(-1) @ hidden
        expected = (context * model.head.out.weight).sum(-1) + model.head.out.bias
        actual = model(pixels, np.ones((1, 3), np.uint8), windows)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
