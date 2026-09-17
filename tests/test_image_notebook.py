import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsnaknee import features, imaging
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256
from rsnaknee.image_notebook import build_image_notebook


def build(tmp_path):
    head = {'feature_names': [f'image_{i:04d}' for i in range(2307)],
            'targets': TARGET_COLUMNS, 'mean': [0.] * 2307, 'scale': [1.] * 2307,
            'coefficients': [[0.] * 2307 for _ in TARGET_COLUMNS],
            'intercepts': list(np.linspace(-1, 1, 12))}
    model = tmp_path / 'model.json'
    model.write_text(json.dumps(head))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'status': 'complete', 'encoder_fit_on_competition_data': False,
        'checkpoint': {'model_type': 'dinov2', 'hidden_size': 384,
                       'files_sha256': {'config.json': 'config-hash', 'pytorch_model.bin': 'weight-hash'}},
        'source_sha256': {name: sha256(Path(module.__file__)) for name, module in
                          [('features.py', features), ('imaging.py', imaging)]},
    }))
    output = tmp_path / 'notebook'
    build_image_notebook(model, manifest, output)
    notebook = json.loads((output / 'image-inference.ipynb').read_text())
    return head, manifest, output, notebook


def portable_namespace(notebook):
    namespace = {'__name__': 'portable_test'}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell['metadata'].get('role') != 'runtime':
            exec(compile(''.join(cell['source']), '<portable notebook>', 'exec'), namespace)
    return namespace


def test_private_offline_notebook_has_only_competition_and_generic_model(tmp_path):
    _, _, output, notebook = build(tmp_path)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True
    assert metadata['enable_gpu'] is True
    assert metadata['enable_internet'] is False
    assert metadata['dataset_sources'] == []
    assert metadata['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert metadata['model_sources'] == ['metaresearch/dinov2/PyTorch/small/1']
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    compile(code, '<notebook>', 'exec')
    assert 'train.csv' not in code
    assert 'train_series.csv' not in code
    assert 'from rsnaknee' not in code
    assert 'weights_only=True' in code
    assert 'local_files_only=True' in code
    assert 'trust_remote_code=False' in code
    portable_namespace(notebook)


def test_1300_dynamic_studies_run_without_training_inputs(tmp_path):
    head, _, _, notebook = build(tmp_path)
    ns = portable_namespace(notebook)
    ids = [f'new-test-{i}' for i in range(1300)][::-1]
    test = pd.DataFrame({ID_COLUMN: ids})
    series = pd.DataFrame({ID_COLUMN: ids, 'SeriesInstanceUID': [f'{x}-series' for x in ids]})
    calls = []
    def prepare(root, split, study, rows):
        assert split == 'test' and rows[0][ID_COLUMN] == study
        calls.append(study)
        return {'images': np.ones((3, 3, 2, 2), dtype=np.float32), 'presence': np.array([1, 0, 1])}
    def encode(pixels):
        return np.ones((len(pixels), 768), dtype=np.float32)
    ns['prepare_study'] = prepare
    result = ns['predict_test_images'](test, series, tmp_path, head, encode)
    assert result[ID_COLUMN].tolist() == ids
    assert len(calls) == 1300
    assert list(result.columns) == [ID_COLUMN, *TARGET_COLUMNS]
    expected = 1 / (1 + np.exp(-np.array(head['intercepts'])))
    np.testing.assert_allclose(result[TARGET_COLUMNS].to_numpy(), np.tile(expected, (1300, 1)))


def test_study_with_no_usable_planes_fails(tmp_path):
    head, _, _, notebook = build(tmp_path)
    ns = portable_namespace(notebook)
    ns['prepare_study'] = lambda *args: {'images': np.zeros((3, 3, 2, 2)), 'presence': np.zeros(3)}
    test = pd.DataFrame({ID_COLUMN: ['test']})
    with pytest.raises(ValueError, match='usable'):
        ns['predict_test_images'](test, pd.DataFrame({ID_COLUMN: ['test']}), tmp_path, head, lambda x: x)


def test_source_changed_since_extraction_is_rejected(tmp_path):
    _, manifest_path, output, _ = build(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest['source_sha256']['imaging.py'] = 'different'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source'):
        build_image_notebook(tmp_path / 'model.json', manifest_path, output.with_name('changed-build'))


def test_checkpoint_hash_mismatch_prevents_loading(tmp_path):
    _, _, _, notebook = build(tmp_path)
    ns = portable_namespace(notebook)
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='checkpoint'):
        ns['verify_checkpoint'](tmp_path, {'files_sha256': {'config.json': 'different'}})


def test_existing_build_is_not_overwritten(tmp_path):
    _, manifest, output, _ = build(tmp_path)
    before = sha256(output / 'image-inference.ipynb')
    with pytest.raises(FileExistsError):
        build_image_notebook(tmp_path / 'model.json', manifest, output)
    assert sha256(output / 'image-inference.ipynb') == before


@pytest.mark.parametrize('hashes', [{}, {'config.json': 'hash'}, {'pytorch_model.bin': 'hash'}])
def test_checkpoint_provenance_must_include_config_and_weights(tmp_path, hashes):
    from rsnaknee.image_notebook import verify_checkpoint
    with pytest.raises(ValueError, match='checkpoint'):
        verify_checkpoint(tmp_path, {'files_sha256': hashes})


def test_unrecorded_checkpoint_weights_are_rejected(tmp_path):
    from rsnaknee.image_notebook import verify_checkpoint
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'pytorch_model.bin').write_bytes(b'official synthetic bytes')
    expected = {'files_sha256': {name: sha256(tmp_path / name) for name in ['config.json', 'pytorch_model.bin']}}
    verify_checkpoint(tmp_path, expected)
    (tmp_path / 'model.safetensors').write_bytes(b'new unverified weights that loading may prefer')
    with pytest.raises(ValueError, match='checkpoint'):
        verify_checkpoint(tmp_path, expected)
