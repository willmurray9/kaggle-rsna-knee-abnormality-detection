import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from rsnaknee import coverage, window_model
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.features import _hash_file
from rsnaknee.window_notebook import build_window_notebook, load_embedded_head, predict_test_images
from test_features import image_study
from test_image_notebook import portable_namespace
from test_window_model import examples


@pytest.fixture
def saved_experiment(tmp_path):
    values, presence, labels = examples()
    targets, weights = window_model.build_supervision(labels)
    model, _ = window_model.fit_head(values, presence, targets, weights)
    model_path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), model_path)
    source_path = tmp_path / 'window_model_source.py'
    source_path.write_bytes(Path(window_model.__file__).read_bytes())
    manifest = {
        'status': 'complete', 'recipe': 'central12-neighbor3-mean10-v1',
        'encoder_frozen': True, 'encoder_fit_on_competition_data': False, 'labels_or_reports_read': False,
        'checkpoint': {'model_type': 'dinov2', 'hidden_size': 384, 'files_sha256': {
            'config.json': 'a' * 64, 'pytorch_model.bin': window_model.GENERIC_WEIGHT_SHA256}},
        'planes': ['Sagittal', 'Coronal', 'Axial'], 'window_order': [list(range(i, i + 3)) for i in range(10)],
        'embedding_layout': 'CLS[384] then mean-patch[384]', 'window_feature_dtype': 'float16',
        'source_sha256': {name: _hash_file(Path(coverage.__file__).with_name(name)) for name in
                          ('__init__.py', 'coverage.py', 'features.py', 'imaging.py')}}
    manifest_path = tmp_path / 'feature_manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    summary = {'status': 'complete', 'recipe': {'targets': TARGET_COLUMNS},
        'feature_manifest': manifest, 'packages': {'torch': torch.__version__},
        'feature_inputs_sha256': {'manifest.json': _hash_file(manifest_path)},
        'source_sha256': {'src/rsnaknee/window_model.py': _hash_file(Path(window_model.__file__))},
        'artifact_sha256': {path.name: _hash_file(path) for path in (model_path, source_path, manifest_path)}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    return model_path, manifest_path, model, values, presence


def notebook_for(saved_experiment, tmp_path):
    model_path, manifest_path, *_ = saved_experiment
    output = tmp_path / 'build'
    build_window_notebook(model_path, manifest_path, output)
    notebook = json.loads((output / 'window-inference.ipynb').read_text())
    return output, notebook, portable_namespace(notebook)


def test_real_trained_torch_state_roundtrips_through_portable_notebook(saved_experiment, tmp_path):
    model_path, _, model, values, presence = saved_experiment
    output, notebook, ns = notebook_for(saved_experiment, tmp_path)
    portable = ns['load_embedded_head'](ns['HEAD_BASE64'], ns['HEAD_SHA256'])
    expected = window_model.predict_windows(model, values, presence)
    np.testing.assert_array_equal(ns['predict_windows'](portable, values, presence), expected)
    assert base64.b64decode(ns['HEAD_BASE64']) == model_path.read_bytes()
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'willmurray99/rsna-knee-coverage-attention'
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert metadata['model_sources'] == ['metaresearch/dinov2/PyTorch/small/1']
    assert metadata['dataset_sources'] == metadata['kernel_sources'] == []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<portable>', 'exec')
    with pytest.raises(FileExistsError):
        build_window_notebook(saved_experiment[0], saved_experiment[1], output)


def test_1300_dynamic_ids_match_cached_float16_cpu_attention_predictions(saved_experiment, tmp_path, monkeypatch):
    _, _, model, _, _ = saved_experiment
    _, _, ns = notebook_for(saved_experiment, tmp_path)
    ids = [f'hidden-{i}' for i in range(1300)][::-1]
    test = pd.DataFrame({ID_COLUMN: ids})
    series = pd.DataFrame({ID_COLUMN: ids, 'SeriesInstanceUID': ids})
    prepared = {'images': np.broadcast_to(np.arange(12, dtype=np.float32)[None, :, None, None] / 12,
                                          (3, 12, 2, 2)), 'presence': np.array([1, 0, 1], dtype=np.uint8)}
    def prepare(root, split, study, rows):
        assert split == 'test' and rows[0][ID_COLUMN] == study
        return prepared
    def encode(pixels):
        return np.arange(768, dtype=np.float32)[None] / 123 + pixels.mean(axis=(1, 2, 3))[:, None]
    ns['prepare_study'] = prepare
    result = ns['predict_test_images'](test, series, tmp_path, model, encode)
    assert result[ID_COLUMN].tolist() == ids and result.shape == (1300, 13)
    _, cache, presence = coverage.encode_studies([prepared], encode)
    expected = window_model.predict_windows(model, np.repeat(cache, 1300, axis=0), np.repeat(presence, 1300, axis=0))
    np.testing.assert_allclose(result[TARGET_COLUMNS], expected, rtol=0, atol=1e-7)
    monkeypatch.setattr('rsnaknee.window_notebook.prepare_study', prepare)
    local = predict_test_images(test.iloc[:3], series.iloc[:3], tmp_path, model, encode)
    assert local[ID_COLUMN].tolist() == ids[:3] and local.shape == (3, 13)
    # Different study batch sizes can differ by float32 roundoff.
    np.testing.assert_allclose(local[TARGET_COLUMNS], result.iloc[:3][TARGET_COLUMNS], rtol=0, atol=1e-7)


def test_portable_real_dicom_preparation_matches_extraction(saved_experiment, tmp_path, image_study):
    _, _, ns = notebook_for(saved_experiment, tmp_path)
    root, series = image_study
    expected = coverage.prepare_study(root, 'train', '1.2', series)
    actual = ns['prepare_study'](root, 'train', '1.2', series)
    np.testing.assert_array_equal(actual['images'], expected['images'])
    assert actual['series'] == expected['series']


@pytest.mark.parametrize('field', ['model', 'training_source', 'coverage_source', 'feature_chain'])
def test_builder_rejects_broken_model_source_and_feature_hash_chain(saved_experiment, tmp_path, field):
    model_path, manifest_path, *_ = saved_experiment
    summary_path = tmp_path / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if field == 'model':
        model_path.write_bytes(model_path.read_bytes() + b'changed')
    elif field == 'training_source':
        summary['source_sha256']['src/rsnaknee/window_model.py'] = 'changed'
    elif field == 'coverage_source':
        manifest = json.loads(manifest_path.read_text())
        manifest['source_sha256']['coverage.py'] = 'changed'
        manifest_path.write_text(json.dumps(manifest))
        summary['feature_manifest'] = manifest
        summary['feature_inputs_sha256']['manifest.json'] = _hash_file(manifest_path)
        summary['artifact_sha256']['feature_manifest.json'] = _hash_file(manifest_path)
    else:
        summary['feature_inputs_sha256']['manifest.json'] = 'changed'
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='hash|source|provenance'):
        build_window_notebook(model_path, manifest_path, tmp_path / 'broken')
    assert not (tmp_path / 'broken').exists()


def test_embedded_bytes_are_verified_before_torch_load_and_nonfinite_heads_reject(saved_experiment):
    model_path, _, model, *_ = saved_experiment
    encoded = base64.b64encode(model_path.read_bytes()).decode()
    with pytest.raises(ValueError, match='hash'):
        load_embedded_head(encoded, 'wrong')
    state = model.state_dict()
    state['query'][0, 0] = float('nan')
    buffer = io.BytesIO()
    torch.save(state, buffer)
    import hashlib
    payload = buffer.getvalue()
    with pytest.raises(ValueError, match='finite'):
        load_embedded_head(base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest())
