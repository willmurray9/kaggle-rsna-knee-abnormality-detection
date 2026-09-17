import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('torch')

from rsnaknee.data import sha256
from rsnaknee.finetune_notebook import build_finetune_notebook


@pytest.fixture
def synthetic_training_repo(tmp_path, monkeypatch):
    from rsnaknee import finetune_notebook
    from rsnaknee.constants import TARGET_COLUMNS
    from test_finetune import examples

    root = tmp_path / 'project'
    package = root / 'src/rsnaknee'
    package.mkdir(parents=True)
    for name in (*finetune_notebook.SOURCES, 'finetune_notebook.py'):
        (package / name).write_bytes(Path(finetune_notebook.__file__).with_name(name).read_bytes())
    labels = examples()[2]
    for target in TARGET_COLUMNS:
        labels[target + '__derived'] = labels[target].map({0: .08, 1: .82})
        labels[target + '__confidence'] = .85
        labels[target + '__source'] = 'synthetic-test'
    inputs = root / 'data/processed/image-v1'
    inputs.mkdir(parents=True)
    labels.to_csv(inputs / 'report_labels.csv')
    labels[['group_id', 'fold']].to_csv(inputs / 'folds.csv')
    audit = root / 'artifacts/reports/label_audit_image_v1.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({
        'label_table_sha256': sha256(inputs / 'report_labels.csv'),
        'folds_sha256': sha256(inputs / 'folds.csv'),
    }))
    monkeypatch.setattr(finetune_notebook, '__file__', str(package / 'finetune_notebook.py'))
    return root


@pytest.mark.parametrize('arms', [None, ('late_blocks', 'deep_blocks'), ('late_blocks', 'soft_targets')])
def test_private_offline_notebook_restores_exact_audited_inputs_and_sources(tmp_path, arms, synthetic_training_repo):
    output = tmp_path / 'build'
    options = {} if arms is None else {'arms': arms}
    expected_arms = ('frozen', 'late_blocks') if arms is None else arms
    build_finetune_notebook(output, **options)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['title'] == ('RSNA Knee Soft Target Training' if arms and 'soft_targets' in arms else 'RSNA Knee Depth Training' if arms else 'RSNA Knee Adaptation Training')
    assert metadata['is_private'] is True and metadata['enable_internet'] is False
    assert metadata['enable_gpu'] is True and metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == ['willmurray99/rsna-knee-coverage-features']
    assert metadata['model_sources'] == ['metaresearch/dinov2/PyTorch/small/1']
    assert metadata['competition_sources'] == ['rsna-knee-abnormality-detection']
    notebook = json.loads((output / metadata['code_file']).read_text())
    code = {c['metadata']['role']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
    for source in code.values():
        compile(source, 'notebook', 'exec')
    working = tmp_path / 'working'
    working.mkdir()
    namespace = {}
    exec(code['bootstrap'].replace("Path('/kaggle/working')", f'Path({str(working)!r})'), namespace)
    root = Path(__file__).resolve().parents[1]
    for name, original in [('report_labels.csv', synthetic_training_repo / 'data/processed/image-v1/report_labels.csv'),
                           ('folds.csv', synthetic_training_repo / 'data/processed/image-v1/folds.csv'),
                           ('label_audit.json', synthetic_training_repo / 'artifacts/reports/label_audit_image_v1.json')]:
        assert sha256(working / 'inputs' / name) == sha256(original)
    assert 'Report' not in pd.read_csv(working / 'inputs/report_labels.csv', nrows=1).columns
    for source in (working / 'rsnaknee').glob('*.py'):
        assert source.read_bytes() == (root / 'src/rsnaknee' / source.name).read_bytes()
    assert namespace['os'].environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert namespace['ARMS'] == expected_arms
    assert namespace['CODE_PROVENANCE'] == {'git_revision': None, 'git_dirty': None}
    manifest = json.loads((output / 'build_manifest.json').read_text())
    assert manifest['selected_arms'] == list(expected_arms)
    assert manifest['code_provenance'] == namespace['CODE_PROVENANCE']
    assert manifest['target_mode_by_arm'] == {arm: 'public_scores' if arm == 'soft_targets' else 'binary' for arm in expected_arms}
    assert 'code_provenance=CODE_PROVENANCE' in code['runtime']
    assert manifest['trainable_blocks_by_arm'] == {arm: {'frozen': 0, 'late_blocks': 2, 'deep_blocks': 6, 'soft_targets': 2}[arm]
                                                 for arm in expected_arms}
    assert manifest['kernel_metadata_sha256'] == sha256(output / 'kernel-metadata.json')
    assert manifest['notebook_sha256'] == sha256(output / metadata['code_file'])
    assert manifest['builder_sha256'] == sha256(root / 'src/rsnaknee/finetune_notebook.py')
    assert manifest['notebook_writer_sha256'] == sha256(root / 'src/rsnaknee/coverage_notebook.py')
    with pytest.raises(FileExistsError):
        build_finetune_notebook(output)


def test_dynamic_test_only_quantization_and_1300_study_ids(tmp_path):
    import numpy as np
    from rsnaknee.finetune_notebook import predict_test_images
    count = 1300
    ids = [f'replaced-{i}' for i in range(count)]
    test = pd.DataFrame({'StudyInstanceUID': ids})
    series = pd.DataFrame({'StudyInstanceUID': ids, 'SeriesInstanceUID': ids})
    recorded = []
    def prepare(root, split, study, rows):
        assert split == 'test'
        return {'images': np.full((3, 12, 2, 2), .5, np.float32), 'presence': np.ones(3, np.uint8)}
    def predict(pixels, presence):
        recorded.append(pixels.copy())
        assert np.all(pixels == 128) and pixels.dtype == np.uint8
        assert np.all(presence == 1)
        return np.full((len(pixels), 12), .75, np.float32)
    result = predict_test_images(test, series, tmp_path, predict, prepare=prepare)
    assert result['StudyInstanceUID'].tolist() == ids
    assert np.all(result.iloc[:, 1:].to_numpy() == .75)
    assert max(len(batch) for batch in recorded) <= 8
    with pytest.raises(ValueError, match='unknown'):
        predict_test_images(test.iloc[:2], series, tmp_path, predict, prepare=prepare)


@pytest.mark.parametrize('arm', ['late_blocks', 'deep_blocks'])
def test_prepared_image_and_cached_uint8_model_predictions_match(arm):
    import numpy as np
    import torch
    from test_finetune import SmallEncoder
    from rsnaknee.finetune import MRIModel, predict_pixels
    from rsnaknee.finetune_notebook import predict_test_images
    rng = np.random.default_rng(96)
    floats = rng.random((3, 3, 12, 4, 4), dtype=np.float32)
    flags = np.ones((3, 3), np.uint8)
    pixels = np.rint(255 * np.clip(floats, 0, 1)).astype(np.uint8)
    model = MRIModel(SmallEncoder(8), arm).eval()
    expected = predict_pixels(model, pixels, flags, np.arange(3))
    test = pd.DataFrame({'StudyInstanceUID': ['test-a', 'test-b', 'test-c']})
    series = test.assign(SeriesInstanceUID='series')
    lookup = dict(zip(test.StudyInstanceUID, range(3)))
    def prepare(root, split, study, rows):
        return {'images': floats[lookup[study]], 'presence': flags[lookup[study]]}
    def predict(batch, presence):
        return predict_pixels(model, batch, presence, np.arange(len(batch)))
    actual = predict_test_images(test, series, Path('/absent'), predict, prepare=prepare)
    np.testing.assert_allclose(actual.iloc[:, 1:], expected, rtol=0, atol=0)


@pytest.mark.parametrize('arm', ['late_blocks', 'deep_blocks'])
def test_inference_builder_attaches_own_weights_and_rejects_unbound_sources(tmp_path, arm):
    from rsnaknee.finetune_notebook import SOURCES, build_inference_notebook
    root = Path(__file__).resolve().parents[1]
    summary = {'status': 'complete', 'arms': {arm: {'final_fit': {'checkpoint_sha256': 'modelhash'}}},
               'source_sha256': {name: sha256(root / 'src/rsnaknee' / name) for name in SOURCES},
               'feature_manifest_sha256': 'featuremanifesthash', 'checkpoint': {'files_sha256': {'config.json': 'config-hash'}}}
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(summary))
    output = tmp_path / 'inference'
    options = {} if arm == 'late_blocks' else {'training_kernel_id': 'willmurray99/rsna-knee-depth-training'}
    build_inference_notebook(path, arm, output, **options)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['kernel_sources'] == [options.get('training_kernel_id', 'willmurray99/rsna-knee-adaptation-training')]
    assert metadata['title'] == ('RSNA Knee Deep Image' if arm == 'deep_blocks' else 'RSNA Knee Adapted Image')
    assert metadata['is_private'] and not metadata['enable_internet']
    notebook = json.loads((output / metadata['code_file']).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'inference', 'exec')
    summary['source_sha256']['coverage.py'] = 'different'
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='source'):
        build_inference_notebook(path, arm, tmp_path / 'bad', **options)


@pytest.mark.parametrize('arm', ['late_blocks', 'deep_blocks', 'soft_targets'])
def test_inference_verifies_attached_model_sources_and_generic_config(tmp_path, arm):
    from rsnaknee.finetune_notebook import SOURCES, verify_inference_inputs
    root = Path(__file__).resolve().parents[1]
    training = tmp_path / 'training'
    (training / arm).mkdir(parents=True)
    (training / 'source').mkdir()
    for name in SOURCES:
        (training / 'source' / name).write_bytes((root / 'src/rsnaknee' / name).read_bytes())
    model_path = training / arm / 'model.pt'
    model_path.write_bytes(b'tensors-only-placeholder-for-hash-boundary')
    config = tmp_path / 'config.json'
    config.write_text('{"model_type":"dinov2","hidden_size":384}')
    checkpoint = {'files_sha256': {'config.json': sha256(config)}}
    feature = {'checkpoint': checkpoint, 'encoder_fit_on_competition_data': False, 'labels_or_reports_read': False,
               'source_sha256': {name: sha256(training / 'source' / name) for name in ('coverage.py', 'imaging.py', 'features.py')}}
    (training / 'feature_manifest.json').write_text(json.dumps(feature))
    summary = {'status': 'complete', 'arms': {arm: {'final_fit': {'checkpoint_sha256': sha256(model_path)}}},
               'source_sha256': {name: sha256(training / 'source' / name) for name in SOURCES},
               'feature_manifest_sha256': sha256(training / 'feature_manifest.json'), 'checkpoint': checkpoint}
    if arm in ('deep_blocks', 'soft_targets'):
        blocks = 6 if arm == 'deep_blocks' else 2
        summary['arms'][arm]['final_fit']['encoder_adaptation'] = {
            'arm': arm, 'encoder_blocks': 12, 'trainable_blocks': blocks,
            'trainable_block_indices': list(range(12 - blocks, 12)), 'final_layernorm_trainable': True,
            'encoder_trainable_parameters': 1000, 'head_trainable_parameters': 100}
    if arm == 'soft_targets':
        summary['recipe'] = {'target_mode_by_arm': {arm: 'public_scores'}}
        summary['arms'][arm]['final_fit'].update(target_mode='public_scores', targets_sha256='a' * 64, weights_sha256='b' * 64)
    (training / 'summary.json').write_text(json.dumps(summary))
    expected = {'summary_sha256': sha256(training / 'summary.json'), 'arm': arm}
    actual, _ = verify_inference_inputs(training, tmp_path, expected)
    assert actual == model_path
    model_path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='model hash'):
        verify_inference_inputs(training, tmp_path, expected)
    model_path.write_bytes(b'tensors-only-placeholder-for-hash-boundary')
    config.write_text('{}')
    with pytest.raises(ValueError, match='architecture hash'):
        verify_inference_inputs(training, tmp_path, expected)
    config.write_text('{"model_type":"dinov2","hidden_size":384}')
    if arm in ('deep_blocks', 'soft_targets'):
        summary['arms'][arm]['final_fit']['encoder_adaptation']['trainable_blocks'] = 99
        (training / 'summary.json').write_text(json.dumps(summary))
        expected['summary_sha256'] = sha256(training / 'summary.json')
        with pytest.raises(ValueError, match='adaptation provenance'):
            verify_inference_inputs(training, tmp_path, expected)
        del summary['arms'][arm]['final_fit']['encoder_adaptation']
        (training / 'summary.json').write_text(json.dumps(summary))
        expected['summary_sha256'] = sha256(training / 'summary.json')
        with pytest.raises(ValueError, match='adaptation provenance'):
            verify_inference_inputs(training, tmp_path, expected)


@pytest.mark.parametrize('arm,blocks', [('late_blocks', 3), ('deep_blocks', 7), ('soft_targets', 3)])
def test_generated_isolated_package_runs_real_dinov2_training_and_inference(tmp_path, arm, blocks, synthetic_training_repo):
    import os
    import subprocess
    import sys
    output = tmp_path / 'build'
    build_finetune_notebook(output, arms=('late_blocks', arm) if arm != 'late_blocks' else ('late_blocks',))
    notebook = json.loads((output / 'adaptation-training.ipynb').read_text())
    bootstrap = ''.join(next(cell for cell in notebook['cells'] if cell['metadata']['role'] == 'bootstrap')['source'])
    working = tmp_path / 'isolated'
    working.mkdir()
    script = bootstrap.replace("Path('/kaggle/working')", f'Path({str(working)!r})') + '''
import numpy as np
import torch
from transformers import Dinov2Config, Dinov2Model
from rsnaknee.finetune import MRIModel, make_optimizer, training_step, load_model_state, predict_pixels
from rsnaknee.image_model import read_frozen_labels
import rsnaknee.finetune as implementation
assert Path(implementation.__file__).parent == package
labels = read_frozen_labels(inputs / 'report_labels.csv', inputs / 'label_audit.json')
assert labels.index.tolist() == [f'study-{i}' for i in range(9)]
encoder = Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=3, num_attention_heads=6,
                                  mlp_ratio=1, image_size=28, patch_size=14))
model = MRIModel(encoder, 'late_blocks').train()
optimizer = make_optimizer(model)
scaler = torch.amp.GradScaler('cuda', enabled=False)
pixels = np.random.default_rng(8).integers(0, 256, (2, 3, 12, 28, 28), dtype=np.uint8)
flags = np.ones((2, 3), np.uint8)
_, targets, weights, _ = implementation.training_partition(labels, 0, target_mode=implementation.TARGET_MODES[model.arm])
assert implementation.TARGET_MODES[model.arm] == ('public_scores' if model.arm == 'soft_targets' else 'binary')
if model.arm == 'soft_targets':
    assert targets[2, 0] == np.float32(.82) and weights[2, 0] == .25
loss = training_step(model, optimizer, scaler, pixels, flags, targets[1:3],
                     weights[1:3], np.array([[0, 1, 2], [7, 8, 9]]))
assert np.isfinite(loss)
assert encoder.encoder.layer[0].attention.attention.query.weight.grad is None
assert encoder.encoder.layer[-1].attention.attention.query.weight.grad is not None
predictions = predict_pixels(model, pixels, flags, np.arange(2))
assert predictions.shape == (2, 12) and np.isfinite(predictions).all()
path = working / 'model.pt'
torch.save(model.state_dict(), path)
load_model_state(model, path)
np.testing.assert_array_equal(predictions, predict_pixels(model, pixels, flags, np.arange(2)))
print('isolated tiny DINOv2 training/inference passed')
'''
    script = script.replace('num_hidden_layers=3', f'num_hidden_layers={blocks}').replace("MRIModel(encoder, 'late_blocks')", f'MRIModel(encoder, {arm!r})')
    env = dict(os.environ, PYTHONPATH=str(working), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    # The embedded package exceeds Linux's limit for a single command argument.
    script_path = working / 'exercise_package.py'
    script_path.write_text(script)
    completed = subprocess.run([sys.executable, str(script_path)], cwd=working, env=env,
                                capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'isolated tiny DINOv2 training/inference passed' in completed.stdout


def test_training_build_records_real_git_revision_and_dirty_state(tmp_path, synthetic_training_repo):
    import subprocess
    root = synthetic_training_repo
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
    git('init')
    git('add', '.')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-m', 'Synthetic fixture')
    clean = tmp_path / 'clean-build'
    build_finetune_notebook(clean, arms=('late_blocks', 'soft_targets'))
    expected = {'git_revision': git('rev-parse', 'HEAD'), 'git_dirty': False}
    assert json.loads((clean / 'build_manifest.json').read_text())['code_provenance'] == expected
    (root / 'uncommitted.txt').write_text('dirty')
    dirty = tmp_path / 'dirty-build'
    build_finetune_notebook(dirty, arms=('late_blocks', 'soft_targets'))
    assert json.loads((dirty / 'build_manifest.json').read_text())['code_provenance'] == dict(expected, git_dirty=True)


def test_soft_inference_requires_target_depth_and_hash_provenance(tmp_path):
    from rsnaknee.finetune_notebook import SOURCES, build_inference_notebook, verify_arm_provenance
    root = Path(__file__).resolve().parents[1]
    arm = 'soft_targets'
    fit = {'checkpoint_sha256': 'c' * 64, 'target_mode': 'public_scores',
           'targets_sha256': 'a' * 64, 'weights_sha256': 'b' * 64,
           'encoder_adaptation': {'arm': arm, 'encoder_blocks': 12, 'trainable_blocks': 2,
                                  'trainable_block_indices': [10, 11], 'final_layernorm_trainable': True}}
    summary = {'status': 'complete', 'arms': {arm: {'final_fit': fit}},
               'recipe': {'target_mode_by_arm': {arm: 'public_scores'}},
               'code_provenance': {'git_revision': 'd' * 40, 'git_dirty': False},
               'source_sha256': {name: sha256(root / 'src/rsnaknee' / name) for name in SOURCES}}
    verify_arm_provenance(summary, arm)
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(summary))
    output = tmp_path / 'soft-inference'
    build_inference_notebook(path, arm, output, training_kernel_id='willmurray99/rsna-knee-soft-target-training')
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    manifest = json.loads((output / 'build_manifest.json').read_text())
    assert metadata['title'] == 'RSNA Knee Soft Target Image'
    assert metadata['kernel_sources'] == ['willmurray99/rsna-knee-soft-target-training']
    assert manifest['target_mode'] == 'public_scores'
    assert manifest['training_code_provenance'] == summary['code_provenance']
    for field, changed in [('target_mode', 'binary'), ('targets_sha256', 'bad'),
                           ('weights_sha256', None), ('encoder_adaptation', None)]:
        original = fit[field]
        fit[field] = changed
        with pytest.raises(ValueError, match='provenance'):
            verify_arm_provenance(summary, arm)
        fit[field] = original
    summary['recipe']['target_mode_by_arm'][arm] = 'binary'
    with pytest.raises(ValueError, match='provenance'):
        verify_arm_provenance(summary, arm)


def test_new_binary_run_cannot_drop_target_provenance():
    from rsnaknee.finetune_notebook import verify_arm_provenance
    summary = {'recipe': {'target_mode_by_arm': {'late_blocks': 'binary'}},
               'arms': {'late_blocks': {'final_fit': {}}}}
    with pytest.raises(ValueError, match='Target mode provenance'):
        verify_arm_provenance(summary, 'late_blocks')
