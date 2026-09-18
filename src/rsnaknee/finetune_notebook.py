"""Package private offline adaptation training with exact audited, report-free inputs."""

import argparse
import base64
import gzip
import inspect
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.coverage_notebook import DISCOVERY, _write_notebook
from rsnaknee.data import sha256
from rsnaknee.finetune import ARMS, TARGET_MODES, TRAINABLE_BLOCKS, TRAIN_WINDOWS_PER_PLANE, selected_arms
from rsnaknee.image_model import read_frozen_labels

SOURCES = ('__init__.py', 'constants.py', 'submission.py', 'data.py', 'baseline.py',
           'image_model.py', 'window_model.py', 'imaging.py', 'features.py', 'coverage.py', 'finetune.py')


def build_finetune_notebook(output: Path,
                            kernel_id: str = 'willmurray99/rsna-knee-adaptation-training', *, arms=ARMS):
    arms = selected_arms(arms)
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                                    check=True, capture_output=True, text=True).stdout.strip())
        code_provenance = {'git_revision': revision, 'git_dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        code_provenance = {'git_revision': None, 'git_dirty': None}
    labels_path = root / 'data/processed/image-v1/report_labels.csv'
    audit_path = root / 'artifacts/reports/label_audit_image_v1.json'
    labels = read_frozen_labels(labels_path, audit_path)
    allowed = {'group_id', 'fold'} | {t + suffix for t in TARGET_COLUMNS
               for suffix in ('', '__observed', '__derived', '__verdict', '__confidence', '__mask', '__source')}
    if set(labels.columns) != allowed:
        raise ValueError('Embed only audited label columns, never reports or extra fields')
    package = Path(__file__).parent
    sources = {name: (package / name).read_text() for name in SOURCES}
    inputs = {'report_labels.csv': labels_path, 'folds.csv': labels_path.with_name('folds.csv'),
              'label_audit.json': audit_path}
    payload = {name: base64.b64encode(gzip.compress(path.read_bytes(), mtime=0)).decode() for name, path in inputs.items()}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    bootstrap = (
        'import os, sys, json, base64, gzip, hashlib\nfrom pathlib import Path\n'
        'os.environ["HF_HUB_OFFLINE"] = "1"\nos.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
        'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n'
        f'ARMS = {arms!r}\nCODE_PROVENANCE = {code_provenance!r}\n'
        "working = Path('/kaggle/working')\n"
        'package = working / "rsnaknee"\npackage.mkdir(exist_ok=True)\n'
        f'sources = {sources!r}\n'
        'for name, source in sources.items():\n    (package / name).write_text(source)\n'
        'sys.path.insert(0, str(working))\n'
        'inputs = working / "inputs"\ninputs.mkdir(exist_ok=True)\n'
        f'payload = {payload!r}\ninput_sha256 = {hashes!r}\n'
        'for name, encoded in payload.items():\n'
        '    content = gzip.decompress(base64.b64decode(encoded))\n'
        '    if hashlib.sha256(content).hexdigest() != input_sha256[name]:\n'
        '        raise ValueError("Embedded input hash differs")\n'
        '    (inputs / name).write_bytes(content)\n'
    )
    runtime = DISCOVERY + '''from rsnaknee.data import sha256
from rsnaknee.finetune import run_comparison
cache_dirs = []
for root, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
    if 'pixels_uint8.npy' in files and 'manifest.json' in files:
        cache_dirs.append(Path(root))
if len(cache_dirs) != 1:
    raise ValueError('Require exactly one private coverage pixel cache')
audit = json.loads((inputs / 'label_audit.json').read_text())
for name in ('train.csv', 'test.csv', 'train_series.csv', 'test_series.csv'):
    if sha256(roots[0] / name) != audit['input_sha256'][name]:
        raise ValueError('Competition metadata changed: ' + name)
result = run_comparison(cache_dirs[0], checkpoints[0], inputs / 'report_labels.csv',
                        working / 'adaptation', inputs / 'label_audit.json', device='cuda', arms=ARMS,
                        code_provenance=CODE_PROVENANCE)
print(json.dumps({'status': result['status'], 'probe': result['probe'],
                  'arms': result['arms'], 'runtime_seconds': result['runtime_seconds']}, indent=2))
'''
    depths = ', '.join(f'{arm}: final {TRAINABLE_BLOCKS[arm]} blocks trainable' for arm in arms)
    title = ('RSNA Knee Multi Window Training' if 'multi_windows' in arms else
             'RSNA Knee Soft Target Training' if 'soft_targets' in arms else
             'RSNA Knee Depth Training' if 'deep_blocks' in arms else 'RSNA Knee Adaptation Training')
    targets = ', '.join(f'{arm}: {TARGET_MODES[arm]}' for arm in arms)
    windows = ', '.join(f'{arm}: {TRAIN_WINDOWS_PER_PLANE[arm]}' for arm in arms)
    _write_notebook(output, kernel_id, title, 'adaptation-training.ipynb', [
        ('markdown', '# Independent MRI adaptation\n\nMatched uint8-pixel arms: ' + depths + '. '
         'Each adapted arm also trains the final LayerNorm; a frozen arm freezes it. '
         'Targets: ' + targets + '. Gold stays binary, public unknowns stay masked. '
         'Training windows per plane: ' + windows + '. Each window has three sampled slices and a '
         '768-dimensional encoder vector; prediction uses all ten windows per plane. Six fixed epochs, '
         'whole-fold exclusion, 58 final-epoch gold OOF predictions. A disposable 64-step probe per arm '
         'must project all selected arms, each with three folds and a final refit, below 7.5 hours. '
         'Private pixel cache stays on Kaggle.\n', 'description'),
        ('code', bootstrap, 'bootstrap'), ('code', runtime, 'runtime')],
        {'source_sha256': {name: sha256(package / name) for name in SOURCES},
         'code_provenance': code_provenance,
         'target_mode_by_arm': {arm: TARGET_MODES[arm] for arm in arms},
         'train_windows_per_plane': {arm: TRAIN_WINDOWS_PER_PLANE[arm] for arm in arms},
         'selected_arms': list(arms), 'trainable_blocks_by_arm': {arm: TRAINABLE_BLOCKS[arm] for arm in arms},
         'inputs_sha256': hashes, 'contains_reports': False})
    metadata_path = output / 'kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['kernel_sources'] = ['willmurray99/rsna-knee-coverage-features']
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    manifest_path = output / 'build_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['kernel_metadata_sha256'] = sha256(metadata_path)
    manifest['notebook_writer_sha256'] = manifest['builder_sha256']
    manifest['builder_sha256'] = sha256(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def predict_test_images(test, series, data_root, predict, *, prepare=None):
    """Prepare dynamic test IDs only; use the exact training cache quantization."""
    from rsnaknee.coverage import prepare_study
    if list(test.columns) != [ID_COLUMN] or test.empty:
        raise ValueError('Require test IDs only')
    ids = test[ID_COLUMN]
    if ids.isna().any() or ids.duplicated().any() or ids.str.strip().eq('').any():
        raise ValueError('Invalid test study IDs')
    if not set(series[ID_COLUMN]).issubset(set(ids)):
        raise ValueError('Test series contain unknown study IDs')
    prepare = prepare_study if prepare is None else prepare
    grouped = {study: frame.to_dict('records') for study, frame in series.groupby(ID_COLUMN)}
    batches = []
    for start in range(0, len(ids), 8):
        studies = ids.iloc[start:start + 8].tolist()
        prepared = [prepare(data_root, 'test', study, grouped.get(study, [])) for study in studies]
        pixels = np.stack([np.rint(255 * np.clip(item['images'], 0, 1)).astype(np.uint8) for item in prepared])
        presence = np.asarray([item['presence'] for item in prepared], dtype=np.uint8)
        values = np.asarray(predict(pixels, presence))
        if values.shape != (len(studies), 12) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid test probabilities')
        batches.append(values)
    result = test.copy()
    result[TARGET_COLUMNS] = np.concatenate(batches)
    return result


def verify_arm_provenance(summary, arm):
    from rsnaknee.finetune import TARGET_MODES, TRAINABLE_BLOCKS, TRAIN_WINDOWS_PER_PLANE

    fit = summary['arms'][arm]['final_fit']
    adaptation = fit.get('encoder_adaptation')
    if adaptation is not None or arm in ('deep_blocks', 'soft_targets', 'multi_windows'):
        blocks = TRAINABLE_BLOCKS[arm]
        total = (adaptation or {}).get('encoder_blocks', 0)
        if (not adaptation or adaptation.get('arm') != arm or total < blocks
                or adaptation.get('trainable_blocks') != blocks
                or adaptation.get('trainable_block_indices') != list(range(total - blocks, total))
                or adaptation.get('final_layernorm_trainable') != bool(blocks)):
            raise ValueError('Encoder adaptation provenance differs from chosen arm')
    if ('target_mode' in fit or 'target_mode_by_arm' in summary.get('recipe', {})
            or arm in ('soft_targets', 'multi_windows')):
        if (fit.get('target_mode') != TARGET_MODES[arm]
                or summary.get('recipe', {}).get('target_mode_by_arm', {}).get(arm) != TARGET_MODES[arm]):
            raise ValueError('Target mode provenance differs from chosen arm')
        for key in ('targets_sha256', 'weights_sha256'):
            digest = fit.get(key)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in '0123456789abcdef' for char in digest)):
                raise ValueError('Target/weight provenance hash missing or invalid')
    if ('train_windows_per_plane' in fit or 'window_schedule_by_arm' in summary
            or isinstance(summary.get('recipe', {}).get('train_windows_per_plane'), dict)
            or arm == 'multi_windows'):
        count = TRAIN_WINDOWS_PER_PLANE[arm]
        counts = summary.get('recipe', {}).get('train_windows_per_plane')
        schedules = summary.get('window_schedule_by_arm')
        schedule = schedules.get(arm, {}) if isinstance(schedules, dict) else {}
        artifact = 'multi_window_indices.npy' if count == 3 else 'window_indices.npy'
        if (not isinstance(counts, dict) or counts.get(arm) != count
                or fit.get('train_windows_per_plane') != count
                or not isinstance(schedule, dict)
                or schedule.get('train_windows_per_plane') != count
                or schedule.get('artifact') != artifact
                or fit.get('window_schedule_sha256') != schedule.get('window_schedule_sha256')):
            raise ValueError('Window schedule provenance differs from chosen arm')
        for digest in (schedule.get('window_schedule_sha256'), summary.get(Path(artifact).stem + '_sha256')):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in '0123456789abcdef' for char in digest)):
                raise ValueError('Window schedule provenance hash missing or invalid')


def verify_inference_inputs(training_dir, checkpoint_dir, expected):
    import hashlib
    from rsnaknee.finetune import TRAINABLE_BLOCKS

    summary_path = training_dir / 'summary.json'
    if sha256(summary_path) != expected['summary_sha256']:
        raise ValueError('Training summary hash differs')
    summary = json.loads(summary_path.read_text())
    arm = expected['arm']
    if (summary.get('status') != 'complete' or arm not in summary.get('arms', {})
            or arm not in TRAINABLE_BLOCKS):
        raise ValueError('Need completed independent adaptation arm')
    verify_arm_provenance(summary, arm)
    schedule = summary.get('window_schedule_by_arm', {}).get(arm)
    if schedule is not None:
        path = training_dir / schedule['artifact']
        if not path.is_file() or sha256(path) != summary[path.stem + '_sha256']:
            raise ValueError('Window schedule provenance file hash differs')
        table = np.load(path, allow_pickle=False)
        if hashlib.sha256(table.tobytes(order='C')).hexdigest() != schedule['window_schedule_sha256']:
            raise ValueError('Window schedule provenance table hash differs')
    model_path = training_dir / arm / 'model.pt'
    if sha256(model_path) != summary['arms'][arm]['final_fit']['checkpoint_sha256']:
        raise ValueError('Chosen independent model hash differs')
    for name in SOURCES:
        if (sha256(training_dir / 'source' / name) != summary['source_sha256'][name]
                or sha256(Path(__file__).parent / name) != summary['source_sha256'][name]):
            raise ValueError('Training and inference source differs: ' + name)
    feature_path = training_dir / 'feature_manifest.json'
    if sha256(feature_path) != summary['feature_manifest_sha256']:
        raise ValueError('Coverage provenance hash differs')
    feature = json.loads(feature_path.read_text())
    if (feature['checkpoint']['files_sha256'] != summary['checkpoint']['files_sha256']
            or feature.get('encoder_fit_on_competition_data') is not False
            or feature.get('labels_or_reports_read') is not False):
        raise ValueError('Coverage must use the identical generic checkpoint')
    for name, digest in feature['source_sha256'].items():
        if digest != summary['source_sha256'].get(name):
            raise ValueError('Coverage preprocessing source differs: ' + name)
    if sha256(checkpoint_dir / 'config.json') != summary['checkpoint']['files_sha256']['config.json']:
        raise ValueError('Generic encoder architecture hash differs')
    return model_path, summary


def run_inference(data_root, checkpoint_dir, training_dir, expected, output_dir):
    import torch
    from transformers import AutoConfig, AutoModel
    from rsnaknee.finetune import MRIModel, load_model_state, predict_pixels
    from rsnaknee.submission import validate_submission

    started = time.perf_counter()
    model_path, summary = verify_inference_inputs(training_dir, checkpoint_dir, expected)
    config = AutoConfig.from_pretrained(str(checkpoint_dir), local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_config(config, trust_remote_code=False, attn_implementation='eager')
    model = load_model_state(MRIModel(encoder, expected['arm']), model_path,
                             provenance=summary['arms'][expected['arm']]['final_fit'].get('encoder_adaptation')).to('cuda')
    test = pd.read_csv(data_root / 'test.csv', dtype=str)
    series = pd.read_csv(data_root / 'test_series.csv', dtype={ID_COLUMN: str, 'SeriesInstanceUID': str})
    def predict(pixels, presence):
        return predict_pixels(model, pixels, presence, np.arange(len(pixels)))
    submission = predict_test_images(test, series, data_root, predict)
    sample = test.copy()
    sample[TARGET_COLUMNS] = .5
    validate_submission(submission, sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / 'submission.csv', index=False)
    return {'status': 'complete', 'studies': len(submission), 'arm': expected['arm'],
            'train_windows_per_plane': summary['arms'][expected['arm']]['final_fit'].get('train_windows_per_plane', 1),
            'window_schedule_sha256': summary['arms'][expected['arm']]['final_fit'].get('window_schedule_sha256'),
            'training_summary_sha256': expected['summary_sha256'],
            'model_sha256': summary['arms'][expected['arm']]['final_fit']['checkpoint_sha256'],
            'runtime_seconds': time.perf_counter() - started,
            'test_sha256': sha256(data_root / 'test.csv'), 'test_series_sha256': sha256(data_root / 'test_series.csv'),
            'submission_sha256': sha256(output_dir / 'submission.csv')}


def build_inference_notebook(summary_path: Path, arm: str, output: Path,
                              kernel_id='willmurray99/rsna-knee-adapted-image', *,
                              training_kernel_id='willmurray99/rsna-knee-adaptation-training'):
    summary = json.loads(summary_path.read_text())
    if (summary.get('status') != 'complete' or arm not in summary.get('arms', {})
            or arm not in TRAINABLE_BLOCKS):
        raise ValueError('Choose one completed independent adaptation arm')
    if arm in ('soft_targets', 'multi_windows') or 'window_schedule_by_arm' in summary:
        verify_arm_provenance(summary, arm)
    package = Path(__file__).parent
    for name in SOURCES:
        if sha256(package / name) != summary.get('source_sha256', {}).get(name):
            raise ValueError('Current source differs from completed training: ' + name)
    sources = {name: (package / name).read_text() for name in SOURCES}
    expected = {'summary_sha256': sha256(summary_path), 'arm': arm}
    helpers = '\n\n'.join(inspect.getsource(function) for function in
                          (predict_test_images, verify_arm_provenance, verify_inference_inputs, run_inference))
    # Helpers live beside the identical training modules so source verification is local.
    sources['adaptation_inference.py'] = (
        'import json, time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\n'
        'from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS\nfrom rsnaknee.data import sha256\n'
        f'SOURCES = {SOURCES!r}\n' + helpers + '\n')
    bootstrap = (
        'import os, sys, json\nfrom pathlib import Path\n'
        'os.environ["HF_HUB_OFFLINE"] = "1"\nos.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
        'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n'
        "working = Path('/kaggle/working')\npackage = working / 'rsnaknee'\npackage.mkdir(exist_ok=True)\n"
        f'sources = {sources!r}\nEXPECTED = {expected!r}\n'
        'for name, source in sources.items():\n    (package / name).write_text(source)\n'
        'sys.path.insert(0, str(working))\n')
    runtime = DISCOVERY + '''from rsnaknee.adaptation_inference import run_inference
training_dirs = []
for root, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
    if 'summary.json' in files and 'feature_manifest.json' in files and (Path(root) / EXPECTED['arm'] / 'model.pt').is_file():
        training_dirs.append(Path(root))
if len(training_dirs) != 1:
    raise ValueError('Require exactly one completed private independent training mount')
result = run_inference(roots[0], checkpoints[0], training_dirs[0], EXPECTED, working)
(working / 'inference_manifest.json').write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
'''
    title = ('RSNA Knee Multi Window Image' if arm == 'multi_windows' else
             'RSNA Knee Soft Target Image' if arm == 'soft_targets' else
             'RSNA Knee Deep Image' if arm == 'deep_blocks' else 'RSNA Knee Adapted Image')
    _write_notebook(output, kernel_id, title, 'adapted-image.ipynb', [
        ('markdown', '# Independent adapted image model\n\nOur own audited generic-initialized weights. '
         'Dynamic test images, identical quantized preprocessing, all thirty windows; '
         'no training images, reports or labels read.\n', 'description'),
        ('code', bootstrap, 'bootstrap'), ('code', runtime, 'runtime')],
        {'source_sha256': summary['source_sha256'], 'training_summary_sha256': sha256(summary_path),
         'training_kernel_id': training_kernel_id, 'target_mode': TARGET_MODES[arm],
         'train_windows_per_plane': TRAIN_WINDOWS_PER_PLANE[arm],
         'window_schedule_sha256': summary['arms'][arm]['final_fit'].get('window_schedule_sha256'),
         'training_code_provenance': summary.get('code_provenance'),
         'arm': arm, 'model_sha256': summary['arms'][arm]['final_fit']['checkpoint_sha256']})
    metadata_path = output / 'kernel-metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['kernel_sources'] = [training_kernel_id]
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    manifest_path = output / 'build_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['kernel_metadata_sha256'] = sha256(metadata_path)
    manifest['notebook_writer_sha256'] = manifest['builder_sha256']
    manifest['builder_sha256'] = sha256(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel-id')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--arm', choices=tuple(TRAINABLE_BLOCKS))
    parser.add_argument('--arms', nargs='+', choices=tuple(TRAINABLE_BLOCKS), default=ARMS)
    parser.add_argument('--training-kernel-id', default='willmurray99/rsna-knee-adaptation-training')
    args = parser.parse_args()
    if args.summary or args.arm:
        if not args.summary or not args.arm:
            parser.error('Inference requires both --summary and --arm')
        build_inference_notebook(args.summary, args.arm, args.output,
                                 args.kernel_id or 'willmurray99/rsna-knee-adapted-image',
                                 training_kernel_id=args.training_kernel_id)
    else:
        build_finetune_notebook(args.output, args.kernel_id or 'willmurray99/rsna-knee-adaptation-training', arms=args.arms)
