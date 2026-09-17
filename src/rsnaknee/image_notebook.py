"""Package a fitted frozen-image head with its exact test-time preprocessing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsnaknee import features, imaging
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256
from rsnaknee.features import _hash_file, assemble_features, normalize_rgb, prepare_study
from rsnaknee.image_model import predict_heads
from rsnaknee.submission import validate_submission


def verify_checkpoint(directory: Path, expected: dict) -> None:
    """Match every source checkpoint file before restricted local loading."""
    root = directory.resolve()
    hashes = expected.get('files_sha256', {})
    weight_suffixes = {'.safetensors', '.bin', '.pt', '.pth'}
    if 'config.json' not in hashes or not any(Path(name).suffix in weight_suffixes for name in hashes):
        raise ValueError('Generic checkpoint provenance requires config and weight hashes')
    actual_files = {str(path.relative_to(root)) for path in root.rglob('*')
                    if path.is_file() and path.suffix in weight_suffixes | {'.json'}}
    if actual_files != set(hashes):
        raise ValueError('Generic checkpoint file inventory differs from extraction')
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _hash_file(path) != digest:
            raise ValueError(f'Generic checkpoint hash mismatch: {name}')


def predict_test_images(test: pd.DataFrame, series: pd.DataFrame, data_root: Path,
                        head: dict, encode, batch_size: int = 12) -> pd.DataFrame:
    """Read test studies in runtime order and apply the frozen extraction recipe."""
    if list(test.columns) != [ID_COLUMN] or test.empty:
        raise ValueError('Invalid test schema or empty test set')
    ids = test[ID_COLUMN]
    if ids.isna().any() or ids.duplicated().any() or ids.str.strip().eq('').any():
        raise ValueError('Test study IDs must be nonempty and unique')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    if not set(series[ID_COLUMN]).issubset(set(ids)):
        raise ValueError('Test series contain an unknown study ID')
    grouped = {study: rows.to_dict('records') for study, rows in series.groupby(ID_COLUMN)}
    predictions = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(test), batch_size):
            batch = ids.iloc[start:start + batch_size].tolist()
            futures = [pool.submit(prepare_study, data_root, 'test', study, grouped.get(study, []))
                       for study in batch]
            prepared = [future.result() for future in futures]
            if any(not item['presence'].any() for item in prepared):
                raise ValueError('Test study has no usable image planes')
            locations = [(i, plane) for i, item in enumerate(prepared)
                         for plane, present in enumerate(item['presence']) if present]
            pixels = normalize_rgb(np.stack([prepared[i]['images'][plane] for i, plane in locations]))
            vectors = assemble_features(encode(pixels), locations, len(prepared))
            if vectors.shape[1] != len(head['feature_names']):
                raise ValueError('Encoder features differ from fitted image head')
            frame = pd.DataFrame(vectors, index=pd.Index(batch, name=ID_COLUMN), columns=head['feature_names'])
            predictions.append(predict_heads(head, frame))
    submission = pd.concat(predictions).reset_index()
    sample = test.copy()
    sample[TARGET_COLUMNS] = 0.5
    validate_submission(submission, sample)
    return submission


def run_image_inference(data_root: Path, checkpoint_dir: Path, head: dict,
                        expected_checkpoint: dict, output: Path) -> dict:
    import time
    import torch
    import transformers
    from transformers import AutoModel

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('This reviewed inference recipe requires the Kaggle GPU')
    verify_checkpoint(checkpoint_dir, expected_checkpoint)
    encoder = AutoModel.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=False, weights_only=True,
    ).to('cuda').eval()
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
               'submission_sha256': _hash_file(output / 'submission.csv')}
    (output / 'inference_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def build_image_notebook(model_path: Path, feature_manifest_path: Path, output: Path,
                         kernel_id: str = 'willmurray99/rsna-knee-independent-image') -> None:
    if output.exists():
        raise FileExistsError(f'Preserve earlier notebook builds: {output}')
    head = json.loads(model_path.read_text())
    manifest = json.loads(feature_manifest_path.read_text())
    if manifest['status'] != 'complete' or manifest['encoder_fit_on_competition_data'] is not False:
        raise ValueError('Require completed generic feature extraction')
    for name, module in [('features.py', features), ('imaging.py', imaging)]:
        if sha256(Path(module.__file__)) != manifest['source_sha256'][name]:
            raise ValueError(f'Preprocessing source changed since extraction: {name}')
    if manifest['checkpoint']['model_type'] != 'dinov2' or manifest['checkpoint']['hidden_size'] != 384:
        raise ValueError('Require the extraction DINOv2-small checkpoint')
    width = 2307
    if head['targets'] != TARGET_COLUMNS or head['feature_names'] != [f'image_{i:04d}' for i in range(width)]:
        raise ValueError('Unexpected image head targets or feature order')
    for name, shape in [('mean', (width,)), ('scale', (width,)), ('coefficients', (12, width)), ('intercepts', (12,))]:
        values = np.asarray(head[name], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid image head {name}')
    if np.any(np.asarray(head['scale']) <= 0):
        raise ValueError('Image head scaling must be positive')

    definitions = (
        'import os\n'
        'os.environ["HF_HUB_OFFLINE"] = "1"\n'
        'os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
        'from collections.abc import Mapping, Sequence\n'
        'from concurrent.futures import ThreadPoolExecutor\n'
        'from pathlib import Path\nfrom typing import Any\nimport hashlib\nimport json\n'
        'import numpy as np\nimport pandas as pd\n'
        'from pandas.api.types import is_complex_dtype, is_numeric_dtype\n'
        f'ID_COLUMN = {ID_COLUMN!r}\nTARGET_COLUMNS = {TARGET_COLUMNS!r}\n'
        f'PLANES = {features.PLANES!r}\nHEADER_TAGS = {features.HEADER_TAGS!r}\n'
        f'HEAD = json.loads({json.dumps(head)!r})\n'
        f'CHECKPOINT = json.loads({json.dumps(manifest["checkpoint"])!r})\n'
        f'HEAD_SHA256 = {sha256(model_path)!r}\n'
    )
    copied = [imaging.order_slices, imaging.decode_grayscale, imaging.preprocess_slice, imaging.choose_series,
              features._hash_file, features._header_hash, features._pixel_spacing, features.prepare_study,
              features.normalize_rgb, features.assemble_features, predict_heads, validate_submission,
              verify_checkpoint, predict_test_images, run_image_inference]
    helpers = '\n\n'.join(inspect.getsource(function) for function in copied)
    runtime = '''roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
         Path('/kaggle/input/rsna-knee-abnormality-detection')]
roots = [root for root in roots if (root / 'test.csv').is_file()]
if len(roots) != 1:
    raise ValueError('Require exactly one competition data mount')
checkpoints = []
for root, dirs, files in os.walk('/kaggle/input'):
    dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
    if 'config.json' in files and 'dinov2' in root.lower():
        config = json.loads((Path(root) / 'config.json').read_text())
        if config.get('model_type') == 'dinov2' and config.get('hidden_size') == 384:
            checkpoints.append(Path(root))
if len(checkpoints) != 1:
    raise ValueError('Require exactly one generic DINOv2-small mount')
result = run_image_inference(roots[0], checkpoints[0], HEAD, CHECKPOINT, Path('/kaggle/working'))
result['head_sha256'] = HEAD_SHA256
Path('/kaggle/working/inference_manifest.json').write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
'''
    cells = []
    for kind, source, role in [
        ('markdown', '# RSNA Knee: independent frozen image model\n\n'
         'Generic DINOv2-small plus fitted regularized heads. Test-only inference uses the exact '
         'saved extraction source and verifies generic checkpoint hashes. Keep private, GPU enabled, '
         'internet disabled. No competition-trained encoder or training reports are required.\n', 'description'),
        ('code', definitions, 'definitions'), ('code', helpers, 'helpers'), ('code', runtime, 'runtime'),
    ]:
        cell = {'cell_type': kind, 'metadata': {'role': role}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    metadata = {'id': kernel_id, 'title': 'RSNA Knee Independent Image', 'code_file': 'image-inference.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': False, 'competition_sources': ['rsna-knee-abnormality-detection'],
                'dataset_sources': [], 'model_sources': ['metaresearch/dinov2/PyTorch/small/1'], 'kernel_sources': []}
    output.mkdir(parents=True, exist_ok=True)
    notebook_path = output / 'image-inference.ipynb'
    notebook_path.write_text(json.dumps(notebook, indent=2) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'build_manifest.json').write_text(json.dumps({
        'model_sha256': sha256(model_path), 'feature_manifest_sha256': sha256(feature_manifest_path),
        'preprocessing_source_sha256': manifest['source_sha256'],
        'checkpoint': manifest['checkpoint'], 'notebook_sha256': sha256(notebook_path),
        'packager_sha256': sha256(Path(__file__)),
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--features-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_image_notebook(args.model, args.features_manifest, args.output)
    print(args.output)
