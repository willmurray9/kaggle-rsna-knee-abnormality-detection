"""Package cached-window attention for private, offline, test-only GPU inference."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsnaknee import coverage, features, imaging, window_model
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.coverage import encode_studies, prepare_study
from rsnaknee.coverage_notebook import DISCOVERY, _write_notebook
from rsnaknee.features import _hash_file
from rsnaknee.image_notebook import verify_checkpoint
from rsnaknee.submission import validate_submission
from rsnaknee.window_model import AttentionHead, predict_windows


def load_embedded_head(encoded: str, expected_sha256: str) -> AttentionHead:
    """Verify exact training bytes, then load only a finite tensor state dictionary."""
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Embedded attention head hash mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a tensor state dictionary')
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Expected named tensor weights only')
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Attention weights must be finite float32 tensors')
    model = AttentionHead().cpu()
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict_test_images(test: pd.DataFrame, series: pd.DataFrame, data_root: Path,
                        head: AttentionHead, encode, batch_size: int = 4) -> pd.DataFrame:
    """Match training's float16 window cache before CPU float32 head prediction."""
    if list(test.columns) != [ID_COLUMN] or test.empty:
        raise ValueError('Invalid test schema or empty test set')
    ids = test[ID_COLUMN]
    if ids.isna().any() or ids.duplicated().any() or ids.str.strip().eq('').any():
        raise ValueError('Test study IDs must be nonempty and unique')
    if not 1 <= batch_size <= 12:
        raise ValueError('Use 1-12 studies per batch')
    if not set(series[ID_COLUMN]).issubset(set(ids)):
        raise ValueError('Test series contain an unknown study ID')
    grouped = {study: rows.to_dict('records') for study, rows in series.groupby(ID_COLUMN)}
    torch.set_num_threads(1)
    head.cpu().eval()
    predictions = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(test), batch_size):
            batch = ids.iloc[start:start + batch_size].tolist()
            futures = [pool.submit(prepare_study, data_root, 'test', study, grouped.get(study, [])) for study in batch]
            prepared = [future.result() for future in futures]
            _, windows, presence = encode_studies(prepared, encode, encoder_batch_size=32)
            # The shared extractor returns float16 windows, exactly as model training reads them.
            probabilities = predict_windows(head, windows, presence.astype(np.uint8))
            frame = pd.DataFrame(probabilities, columns=TARGET_COLUMNS)
            frame.insert(0, ID_COLUMN, batch)
            predictions.append(frame)
    submission = pd.concat(predictions, ignore_index=True)
    sample = test.copy()
    sample[TARGET_COLUMNS] = .5
    validate_submission(submission, sample)
    return submission


def run_window_inference(data_root: Path, checkpoint_dir: Path, head: AttentionHead,
                         expected_checkpoint: dict, output: Path) -> dict:
    import time
    import transformers
    from transformers import AutoModel

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('Window attention inference requires the Kaggle GPU')
    verify_checkpoint(checkpoint_dir, expected_checkpoint)
    encoder = AutoModel.from_pretrained(str(checkpoint_dir), local_files_only=True,
                                       trust_remote_code=False, weights_only=True).to('cuda').float().eval()
    encoder.requires_grad_(False)
    torch.manual_seed(0)

    def encode(pixels):
        with torch.no_grad():
            tokens = encoder(pixel_values=torch.from_numpy(pixels).to('cuda')).last_hidden_state
            return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1).float().cpu().numpy()

    test = pd.read_csv(data_root / 'test.csv', dtype={ID_COLUMN: str})
    series = pd.read_csv(data_root / 'test_series.csv', dtype={ID_COLUMN: str, 'SeriesInstanceUID': str})
    submission = predict_test_images(test, series, data_root, head, encode)
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / 'submission.csv', index=False)
    summary = {'studies': len(submission), 'runtime_seconds': time.perf_counter() - started,
               'torch': torch.__version__, 'transformers': transformers.__version__,
               'gpu': torch.cuda.get_device_name(0), 'checkpoint_verified': True,
               'encoder_dtype': 'float32', 'window_cache_dtype': 'float16', 'head_device': 'cpu',
               'head_dtype': 'float32', 'recipe': 'central12-neighbor3-mean10-v1',
               'input_sha256': {name: _hash_file(data_root / name) for name in ('test.csv', 'test_series.csv')},
               'submission_sha256': _hash_file(output / 'submission.csv')}
    (output / 'inference_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def build_window_notebook(model_path: Path, feature_manifest_path: Path, output: Path,
                          kernel_id: str = 'willmurray99/rsna-knee-coverage-attention') -> None:
    """Bind the fitted model, training source and exact coverage extraction manifest."""
    model_path, feature_manifest_path, output = map(Path, (model_path, feature_manifest_path, output))
    if output.exists():
        raise FileExistsError('Preserve earlier notebook builds')
    summary_path = model_path.with_name('summary.json')
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(feature_manifest_path.read_text())
    if summary.get('status') != 'complete' or manifest.get('status') != 'complete':
        raise ValueError('Require completed attention training and extraction')
    if summary.get('recipe', {}).get('targets') != TARGET_COLUMNS:
        raise ValueError('Training target order differs')
    model_sha = _hash_file(model_path)
    manifest_sha = _hash_file(feature_manifest_path)
    if (summary.get('artifact_sha256', {}).get(model_path.name) != model_sha
            or summary.get('feature_inputs_sha256', {}).get('manifest.json') != manifest_sha
            or summary.get('artifact_sha256', {}).get('feature_manifest.json') != manifest_sha
            or summary.get('feature_manifest') != manifest):
        raise ValueError('Model/feature manifest provenance hash chain differs')
    saved_manifest = model_path.with_name('feature_manifest.json')
    if not saved_manifest.is_file() or _hash_file(saved_manifest) != manifest_sha:
        raise ValueError('Saved training feature manifest hash differs')
    training_source_sha = _hash_file(Path(window_model.__file__))
    saved_source = model_path.with_name('window_model_source.py')
    if (summary.get('source_sha256', {}).get('src/rsnaknee/window_model.py') != training_source_sha
            or not saved_source.is_file() or _hash_file(saved_source) != training_source_sha
            or summary.get('artifact_sha256', {}).get('window_model_source.py') != training_source_sha):
        raise ValueError('Attention source differs from training provenance')
    if (manifest.get('encoder_frozen') is not True or manifest.get('encoder_fit_on_competition_data') is not False
            or manifest.get('labels_or_reports_read') is not False):
        raise ValueError('Require frozen generic extraction without report/label input')
    if (manifest.get('recipe') != 'central12-neighbor3-mean10-v1'
            or manifest.get('planes') != list(features.PLANES)
            or manifest.get('window_order') != [list(range(i, i + 3)) for i in range(10)]
            or manifest.get('embedding_layout') != 'CLS[384] then mean-patch[384]'
            or manifest.get('window_feature_dtype') != 'float16'):
        raise ValueError('Window extraction layout differs')
    for name in ('__init__.py', 'coverage.py', 'features.py', 'imaging.py'):
        if _hash_file(Path(__file__).with_name(name)) != manifest.get('source_sha256', {}).get(name):
            raise ValueError(f'Preprocessing source changed since extraction: {name}')
    checkpoint = manifest.get('checkpoint', {})
    hashes = checkpoint.get('files_sha256', {})
    if (checkpoint.get('model_type') != 'dinov2' or checkpoint.get('hidden_size') != 384
            or hashes.get('pytorch_model.bin') != window_model.GENERIC_WEIGHT_SHA256
            or 'config.json' not in hashes):
        raise ValueError('Require the audited generic DINOv2-small checkpoint provenance')
    encoded = base64.b64encode(model_path.read_bytes()).decode('ascii')
    load_embedded_head(encoded, model_sha)
    source_hashes = {**manifest['source_sha256'], 'window_model.py': training_source_sha,
                     'window_notebook.py': _hash_file(Path(__file__)),
                     'image_notebook.py': _hash_file(Path(__file__).with_name('image_notebook.py')),
                     'submission.py': _hash_file(Path(__file__).with_name('submission.py'))}
    definitions = (
        'import os\nos.environ["HF_HUB_OFFLINE"]="1"\nos.environ["TRANSFORMERS_OFFLINE"]="1"\n'
        'from collections.abc import Mapping, Sequence\nfrom concurrent.futures import ThreadPoolExecutor\n'
        'from pathlib import Path\nfrom typing import Any\nimport base64\nimport hashlib\nimport io\nimport json\n'
        'import numpy as np\nimport pandas as pd\nimport torch\nfrom torch import nn\n'
        'from pandas.api.types import is_complex_dtype, is_numeric_dtype\n'
        f'ID_COLUMN={ID_COLUMN!r}\nTARGET_COLUMNS={TARGET_COLUMNS!r}\n'
        f'PLANES={features.PLANES!r}\nHEADER_TAGS={features.HEADER_TAGS!r}\n'
        f'BATCH_SIZE={window_model.BATCH_SIZE!r}\nHEAD_BASE64={encoded!r}\nHEAD_SHA256={model_sha!r}\n'
        f'CHECKPOINT=json.loads({json.dumps(checkpoint)!r})\n'
        f'FEATURE_MANIFEST_SHA256={manifest_sha!r}\nSOURCE_SHA256={source_hashes!r}\n'
        f'TRAINING_TORCH_VERSION={summary.get("packages", {}).get("torch")!r}\n'
        f'EXPERIMENT_SUMMARY_SHA256={_hash_file(summary_path)!r}\n'
    )
    copied = [imaging.order_slices, imaging.decode_grayscale, imaging.preprocess_slice, imaging.choose_series,
              features._hash_file, features._header_hash, features._pixel_spacing, coverage.prepare_study,
              features.normalize_rgb, coverage.encode_studies, window_model.AttentionHead,
              window_model._check_arrays, window_model.predict_windows, load_embedded_head,
              validate_submission, verify_checkpoint, predict_test_images, run_window_inference]
    helpers = '\n\n'.join(inspect.getsource(item) for item in copied)
    runtime = DISCOVERY + '''head = load_embedded_head(HEAD_BASE64, HEAD_SHA256)
result = run_window_inference(roots[0], checkpoints[0], head, CHECKPOINT, Path('/kaggle/working'))
result.update(head_sha256=HEAD_SHA256, feature_manifest_sha256=FEATURE_MANIFEST_SHA256,
              source_sha256=SOURCE_SHA256, training_torch_version=TRAINING_TORCH_VERSION,
              experiment_summary_sha256=EXPERIMENT_SUMMARY_SHA256)
Path('/kaggle/working/inference_manifest.json').write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
'''
    _write_notebook(output, kernel_id, 'RSNA Knee Coverage Attention', 'window-inference.ipynb', [
        ('markdown', '# Independent coverage attention\n\nFrozen generic DINOv2-small and a trained '
         'diagnosis attention head. Exact coverage helpers and float16 window quantization; CPU float32 head. '
         'Checkpoint and embedded head bytes are verified before restricted loading. Test-only runtime.\n', 'description'),
        ('code', definitions, 'definitions'), ('code', helpers, 'helpers'), ('code', runtime, 'runtime')],
        {'model_sha256': model_sha, 'feature_manifest_sha256': manifest_sha,
         'experiment_summary_sha256': _hash_file(summary_path), 'source_sha256': source_hashes,
         'checkpoint': checkpoint, 'training_torch_version': summary.get('packages', {}).get('torch'),
         'packager_sha256': _hash_file(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--features-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel-id', default='willmurray99/rsna-knee-coverage-attention')
    args = parser.parse_args()
    build_window_notebook(args.model, args.features_manifest, args.output, args.kernel_id)
