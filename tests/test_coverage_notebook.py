import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsnaknee import coverage
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.features import _hash_file
from rsnaknee.coverage_notebook import build_coverage_notebook, build_inference_notebook, predict_test_images
from test_image_notebook import portable_namespace
from test_features import image_study


def build_inference(tmp_path):
    head = {'feature_names': [f'image_{i:04d}' for i in range(2307)], 'targets': TARGET_COLUMNS,
            'mean': [0.] * 2307, 'scale': [1.] * 2307, 'coefficients': [[0.] * 2307 for _ in TARGET_COLUMNS],
            'intercepts': list(np.linspace(-1, 1, 12))}
    for i in range(12):
        head['coefficients'][i][i] = 0.1
    model = tmp_path / 'model.json'
    model.write_text(json.dumps(head))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'status': 'complete', 'recipe': 'central12-neighbor3-mean10-v1',
        'encoder_fit_on_competition_data': False, 'encoder_frozen': True,
        'checkpoint': {'model_type': 'dinov2', 'hidden_size': 384,
                       'files_sha256': {'config.json': 'a' * 64, 'pytorch_model.bin': 'b' * 64}},
        'source_sha256': {name: _hash_file(Path(coverage.__file__).with_name(name)) for name in
                          ('__init__.py', 'coverage.py', 'features.py', 'imaging.py')}}))
    (tmp_path / 'summary.json').write_text(json.dumps({
        'artifact_sha256': {'model.json': _hash_file(model)},
        'feature_inputs_sha256': {'manifest.json': _hash_file(manifest)}}))
    output = tmp_path / 'inference'
    build_inference_notebook(model, manifest, output)
    return head, manifest, output, json.loads((output / 'coverage-inference.ipynb').read_text())


def test_extraction_notebook_materializes_exact_source_and_private_offline_metadata(tmp_path, monkeypatch):
    output = tmp_path / 'build'
    build_coverage_notebook(output)
    notebook = json.loads((output / 'coverage-features.ipynb').read_text())
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'willmurray99/rsna-knee-coverage-features'
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['model_sources'] == ['metaresearch/dinov2/PyTorch/small/1']
    assert metadata['dataset_sources'] == metadata['kernel_sources'] == []
    work = tmp_path / 'work'
    work.mkdir()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'notebook', 'exec')
    import sys
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '0')
    exec(''.join(notebook['cells'][1]['source']).replace('/kaggle/working', str(work)), {})
    for path in (work / 'rsnaknee').glob('*.py'):
        assert path.read_bytes() == Path(coverage.__file__).with_name(path.name).read_bytes()
    with pytest.raises(FileExistsError):
        build_coverage_notebook(output)


def test_portable_inference_processes_1300_dynamic_test_ids_with_feature_parity(tmp_path, monkeypatch):
    head, _, _, notebook = build_inference(tmp_path)
    ns = portable_namespace(notebook)
    ids = [f'test-{i}' for i in range(1300)][::-1]
    test = pd.DataFrame({ID_COLUMN: ids})
    series = pd.DataFrame({ID_COLUMN: ids, 'SeriesInstanceUID': ids})
    def prepare(root, split, study, rows):
        assert split == 'test' and rows[0][ID_COLUMN] == study
        pixels = np.broadcast_to(np.arange(12, dtype=np.float32)[None, :, None, None] / 12, (3, 12, 2, 2))
        return {'images': pixels, 'presence': np.array([1, 0, 1])}
    def encode(pixels):
        return np.repeat(pixels.mean(axis=(1, 2, 3))[:, None], 768, axis=1)
    ns['prepare_study'] = prepare
    result = ns['predict_test_images'](test, series, tmp_path, head, encode)
    monkeypatch.setattr('rsnaknee.coverage_notebook.prepare_study', prepare)
    local = predict_test_images(test, series, tmp_path, head, encode)
    pd.testing.assert_frame_equal(result, local)
    assert result[ID_COLUMN].tolist() == ids
    # Explicit pooled intensity: each window's channel indices average 4.5,5.5,6.5.
    mean = np.mean((np.array([4.5, 5.5, 6.5]) / 12 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
    expected = 1 / (1 + np.exp(-(np.asarray(head['intercepts']) + .1 * mean)))
    np.testing.assert_allclose(result[TARGET_COLUMNS], np.tile(expected, (1300, 1)), atol=1e-7)


def test_inference_refuses_wrong_experiment_or_changed_source(tmp_path):
    _, manifest, output, _ = build_inference(tmp_path)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    summary['feature_inputs_sha256']['manifest.json'] = 'wrong'
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='provenance'):
        build_inference_notebook(tmp_path / 'model.json', manifest, output.with_name('wrong'))
    payload = json.loads(manifest.read_text())
    payload['source_sha256']['coverage.py'] = 'changed'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='source'):
        build_inference_notebook(tmp_path / 'model.json', manifest, output.with_name('changed'))


def test_portable_checkpoint_verification_rejects_changed_weight(tmp_path):
    _, _, _, notebook = build_inference(tmp_path)
    ns = portable_namespace(notebook)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    (checkpoint / 'model.safetensors').write_bytes(b'original')
    hashes = {path.name: _hash_file(path) for path in checkpoint.iterdir()}
    ns['verify_checkpoint'](checkpoint, {'files_sha256': hashes})
    (checkpoint / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        ns['verify_checkpoint'](checkpoint, {'files_sha256': hashes})


def test_portable_prepare_matches_real_dicom_geometry_and_pixel_arrays(tmp_path, image_study):
    root, series = image_study
    _, _, _, notebook = build_inference(tmp_path)
    ns = portable_namespace(notebook)
    expected = coverage.prepare_study(root, 'train', '1.2', series)
    actual = ns['prepare_study'](root, 'train', '1.2', series)
    np.testing.assert_array_equal(actual['images'], expected['images'])
    np.testing.assert_array_equal(actual['presence'], expected['presence'])
    assert actual['series'] == expected['series']
