"""Build private offline coverage extraction and matched mean-head inference notebooks."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsnaknee import coverage, features, imaging
from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.coverage import prepare_study, encode_studies
from rsnaknee.features import _hash_file
from rsnaknee.image_model import predict_heads
from rsnaknee.image_notebook import verify_checkpoint
from rsnaknee.submission import validate_submission


DISCOVERY = '''roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
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
'''


def _write_notebook(output: Path, kernel_id: str, title: str, filename: str,
                    cells: list[tuple[str, str, str]], provenance: dict) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier notebook builds')
    formatted = []
    for kind, source, role in cells:
        cell = {'cell_type': kind, 'metadata': {'role': role}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            compile(source, 'coverage-notebook', 'exec')
            cell.update(execution_count=None, outputs=[])
        formatted.append(cell)
    notebook = {'cells': formatted, 'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    metadata = {'id': kernel_id, 'title': title, 'code_file': filename,
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
                'competition_sources': ['rsna-knee-abnormality-detection'], 'dataset_sources': [],
                'model_sources': ['metaresearch/dinov2/PyTorch/small/1'], 'kernel_sources': []}
    output.mkdir(parents=True)
    (output / filename).write_text(json.dumps(notebook, indent=2) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    provenance.update(notebook_sha256=_hash_file(output / filename),
                      kernel_metadata_sha256=_hash_file(output / 'kernel-metadata.json'),
                      builder_sha256=_hash_file(Path(__file__)),
                      execution='Build only; no upload or execution performed.')
    (output / 'build_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')


def build_coverage_notebook(output: Path, kernel_id: str = 'willmurray99/rsna-knee-coverage-features') -> None:
    package = Path(__file__).parent
    sources = {name: (package / name).read_text() for name in ('__init__.py', 'imaging.py', 'features.py', 'coverage.py')}
    bootstrap = (
        'import os, sys, json\nfrom pathlib import Path\n'
        'os.environ["HF_HUB_OFFLINE"] = "1"\n'
        'os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
        f'sources = {sources!r}\n'
        'package = Path("/kaggle/working/rsnaknee")\n'
        'package.mkdir(exist_ok=True)\n'
        'for name, source in sources.items():\n'
        '    (package / name).write_text(source)\n'
        'sys.path.insert(0, "/kaggle/working")\n'
    )
    runtime = DISCOVERY + '''from rsnaknee.coverage import extract_features
result = extract_features(roots[0], checkpoints[0], Path('/kaggle/working/features'),
                          batch_size=4, workers=4, encoder_batch_size=32, device='cuda')
print(json.dumps({'status': result['status'], 'completed_studies': result['completed_studies'],
                  'elapsed_seconds': result['elapsed_seconds']}, indent=2))
'''
    _write_notebook(output, kernel_id, 'RSNA Knee Coverage Features', 'coverage-features.ipynb', [
        ('markdown', '# Neighboring-slice coverage\n\nFrozen generic DINOv2-small; same selected planes and '
         '224-pixel preprocessing, twelve central slices and ten overlapping RGB windows per plane. '
         'Mean embeddings and window cache are streamed. A separate quantized pixel cache stays private on Kaggle. '
         'No report or condition values are parsed. First-batch timing estimates total runtime.\n', 'description'),
        ('code', bootstrap, 'bootstrap'), ('code', runtime, 'runtime')],
        {'source_sha256': {name: _hash_file(package / name) for name in sources}})


def predict_test_images(test: pd.DataFrame, series: pd.DataFrame, data_root: Path,
                        head: dict, encode, batch_size: int = 4) -> pd.DataFrame:
    """Apply the exact float32 coverage extraction helper to dynamic test IDs."""
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
    predictions = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(test), batch_size):
            batch = ids.iloc[start:start + batch_size].tolist()
            futures = [pool.submit(prepare_study, data_root, 'test', study, grouped.get(study, [])) for study in batch]
            prepared = [future.result() for future in futures]
            vectors, _, _ = encode_studies(prepared, encode, encoder_batch_size=32)
            if vectors.shape[1] != len(head['feature_names']):
                raise ValueError('Encoder features differ from fitted image head')
            frame = pd.DataFrame(vectors, index=pd.Index(batch, name=ID_COLUMN), columns=head['feature_names'])
            predictions.append(predict_heads(head, frame))
    submission = pd.concat(predictions).reset_index()
    sample = test.copy()
    sample[TARGET_COLUMNS] = 0.5
    validate_submission(submission, sample)
    return submission


def run_coverage_inference(data_root: Path, checkpoint_dir: Path, head: dict,
                           expected_checkpoint: dict, output: Path) -> dict:
    import time
    import torch
    import transformers
    from transformers import AutoModel

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('Coverage inference requires the Kaggle GPU')
    verify_checkpoint(checkpoint_dir, expected_checkpoint)
    encoder = AutoModel.from_pretrained(str(checkpoint_dir), local_files_only=True,
                                       trust_remote_code=False, weights_only=True).to('cuda').eval()
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
               'recipe': 'central12-neighbor3-mean10-v1',
               'input_sha256': {name: _hash_file(data_root / name) for name in ('test.csv', 'test_series.csv')},
               'submission_sha256': _hash_file(output / 'submission.csv')}
    (output / 'inference_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def build_inference_notebook(model_path: Path, feature_manifest_path: Path, output: Path,
                             kernel_id: str = 'willmurray99/rsna-knee-coverage-image') -> None:
    """Require source and experiment hashes so baseline heads cannot attach by mistake."""
    model_path, feature_manifest_path, output = map(Path, (model_path, feature_manifest_path, output))
    if output.exists():
        raise FileExistsError('Preserve earlier notebook builds')
    head = json.loads(model_path.read_text())
    manifest = json.loads(feature_manifest_path.read_text())
    if (manifest['status'] != 'complete' or manifest.get('recipe') != 'central12-neighbor3-mean10-v1'
            or manifest['encoder_fit_on_competition_data'] is not False or manifest.get('encoder_frozen') is not True):
        raise ValueError('Require completed frozen coverage extraction')
    for name in ('__init__.py', 'coverage.py', 'features.py', 'imaging.py'):
        if _hash_file(Path(__file__).with_name(name)) != manifest['source_sha256'][name]:
            raise ValueError(f'Preprocessing source changed since extraction: {name}')
    summary = json.loads(model_path.with_name('summary.json').read_text())
    if (summary['artifact_sha256'].get(model_path.name) != _hash_file(model_path)
            or summary['feature_inputs_sha256'].get('manifest.json') != _hash_file(feature_manifest_path)):
        raise ValueError('Head and feature manifest provenance differ from saved experiment')
    checkpoint = manifest['checkpoint']
    if checkpoint['model_type'] != 'dinov2' or checkpoint['hidden_size'] != 384:
        raise ValueError('Require generic DINOv2-small checkpoint')
    hashes = checkpoint.get('files_sha256', {})
    if 'config.json' not in hashes or not any(Path(name).suffix in {'.safetensors', '.bin', '.pt', '.pth'} for name in hashes):
        raise ValueError('Checkpoint provenance requires config and weights')
    if head['targets'] != TARGET_COLUMNS or head['feature_names'] != [f'image_{i:04d}' for i in range(2307)]:
        raise ValueError('Unexpected head targets or feature order')
    for name, shape in [('mean', (2307,)), ('scale', (2307,)), ('coefficients', (12, 2307)), ('intercepts', (12,))]:
        values = np.asarray(head[name], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Invalid head {name}')
    if np.any(np.asarray(head['scale']) <= 0):
        raise ValueError('Head scale must be positive')
    definitions = (
        'import os\nos.environ["HF_HUB_OFFLINE"]="1"\nos.environ["TRANSFORMERS_OFFLINE"]="1"\n'
        'from collections.abc import Mapping, Sequence\nfrom concurrent.futures import ThreadPoolExecutor\n'
        'from pathlib import Path\nfrom typing import Any\nimport hashlib\nimport json\n'
        'import numpy as np\nimport pandas as pd\nfrom pandas.api.types import is_complex_dtype, is_numeric_dtype\n'
        f'ID_COLUMN={ID_COLUMN!r}\nTARGET_COLUMNS={TARGET_COLUMNS!r}\n'
        f'PLANES={features.PLANES!r}\nHEADER_TAGS={features.HEADER_TAGS!r}\n'
        f'HEAD=json.loads({json.dumps(head)!r})\nCHECKPOINT=json.loads({json.dumps(checkpoint)!r})\n'
        f'HEAD_SHA256={_hash_file(model_path)!r}\nFEATURE_MANIFEST_SHA256={_hash_file(feature_manifest_path)!r}\n'
        f'SOURCE_SHA256={manifest["source_sha256"]!r}\n'
    )
    copied = [imaging.order_slices, imaging.decode_grayscale, imaging.preprocess_slice, imaging.choose_series,
              features._hash_file, features._header_hash, features._pixel_spacing, coverage.prepare_study,
              features.normalize_rgb, coverage.encode_studies, predict_heads, validate_submission,
              verify_checkpoint, predict_test_images, run_coverage_inference]
    helpers = '\n\n'.join(inspect.getsource(function) for function in copied)
    runtime = DISCOVERY + '''result = run_coverage_inference(roots[0], checkpoints[0], HEAD, CHECKPOINT, Path('/kaggle/working'))
result.update(head_sha256=HEAD_SHA256, feature_manifest_sha256=FEATURE_MANIFEST_SHA256, source_sha256=SOURCE_SHA256)
Path('/kaggle/working/inference_manifest.json').write_text(json.dumps(result, indent=2) + '\\n')
print(json.dumps(result, indent=2))
'''
    _write_notebook(output, kernel_id, 'RSNA Knee Coverage Image', 'coverage-inference.ipynb', [
        ('markdown', '# Independent coverage model\n\nGeneric frozen DINOv2-small and fitted regularized heads. '
         'Exact extraction helpers, verified checkpoint, dynamic test IDs, no training inputs.\n', 'description'),
        ('code', definitions, 'definitions'), ('code', helpers, 'helpers'), ('code', runtime, 'runtime')],
        {'model_sha256': _hash_file(model_path), 'feature_manifest_sha256': _hash_file(feature_manifest_path),
         'experiment_summary_sha256': _hash_file(model_path.with_name('summary.json')),
         'source_sha256': manifest['source_sha256'], 'checkpoint': checkpoint})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kernel-id')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--features-manifest', type=Path)
    args = parser.parse_args()
    if args.model or args.features_manifest:
        if not args.model or not args.features_manifest:
            parser.error('Inference requires both --model and --features-manifest')
        build_inference_notebook(args.model, args.features_manifest, args.output,
                                 args.kernel_id or 'willmurray99/rsna-knee-coverage-image')
    else:
        build_coverage_notebook(args.output, args.kernel_id or 'willmurray99/rsna-knee-coverage-features')
