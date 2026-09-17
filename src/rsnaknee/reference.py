"""Package the audited public MRI reference for private, offline Kaggle inference."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA256 = 'b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8'
MANIFEST_SHA256 = '496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844'


def strict_submission(pred, studies, test_df, path, targets):
    """Validate before writing; preserve the reference's rank transformation."""
    ids = pd.Series(list(studies), dtype='object')
    expected = test_df['StudyInstanceUID']
    values = np.asarray(pred)
    if expected.empty or expected.isna().any() or expected.duplicated().any():
        raise ValueError('Invalid test IDs')
    if ids.isna().any() or ids.duplicated().any() or set(ids) != set(expected):
        raise ValueError('Prediction IDs do not match test IDs exactly')
    if values.shape != (len(ids), len(targets)):
        raise ValueError('Wrong prediction shape')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid prediction probabilities')
    sub = pd.DataFrame(values, columns=targets).rank(pct=True)
    sub.insert(0, 'StudyInstanceUID', ids.to_numpy())
    sub = sub.set_index('StudyInstanceUID').loc[expected].reset_index()
    sub.to_csv(path, index=False)
    return sub


def checked_replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one audited source fragment: {old[:80]!r}')
    return text.replace(old, new, 1)


RUNTIME_HELPERS = '''
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def single_directory(candidates, description):
    candidates = sorted(set(p.resolve() for p in candidates))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one {description}; found {len(candidates)}')
    return candidates[0]


def find_dinov2(variant='small'):
    if variant != 'small':
        raise ValueError('Only reviewed DINOv2-small is allowed')
    hits = []
    for root, dirs, files in os.walk('/kaggle/input'):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if 'config.json' in files and 'dinov2' in root.lower():
            config = json.loads((Path(root) / 'config.json').read_text())
            if config.get('model_type') == 'dinov2' and config.get('hidden_size') == 384:
                hits.append(Path(root))
    return single_directory(hits, 'DINOv2-small model directory')


def checked_package():
    hits = []
    for root, dirs, files in os.walk('/kaggle/input'):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if 'manifest.json' in files:
            path = Path(root) / 'manifest.json'
            if file_sha256(path) == EXPECTED_MANIFEST_SHA256:
                hits.append(Path(root))
    package = single_directory(hits, 'reviewed weights package')
    manifest = json.loads((package / 'manifest.json').read_text())
    if manifest['targets'] != TARGETS or len(manifest['members']) != 20:
        raise ValueError('Unexpected targets or member count')
    for member in manifest['members']:
        file = (package / member['file']).resolve()
        if file.parent != package or not file.is_file():
            raise ValueError('Missing or unsafe checkpoint path')
    return package


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


def write_submission(pred, studies, test_df, path):
    if len(RUN_RECORD['members']) != 20:
        raise RuntimeError('All 20 reviewed members must finish')
    return strict_submission(pred, studies, test_df, path, TARGETS)


_original_build_cache = build_cache

def build_cache(slot_map, plane_map, lat_map, tag):
    expected = pd.read_csv(ROOT / 'test.csv')['StudyInstanceUID']
    if expected.duplicated().any() or set(slot_map) != set(expected):
        raise ValueError('Image study coverage differs from test IDs')
    needed = len(slot_map) * N_SLOT * CACHE_SLICES * IMG * IMG
    with open('/proc/meminfo') as handle:
        memory = dict(line.split(':', 1) for line in handle if ':' in line)
    available = int(memory['MemAvailable'].split()[0]) * 1024
    if needed > available * 0.70:
        raise RuntimeError('Pixel cache leaves insufficient measured memory headroom')
    studies, cache, mask = _original_build_cache(slot_map, plane_map, lat_map, tag)
    if not np.isfinite(mask).all() or (mask.sum(1) == 0).any():
        raise RuntimeError('At least one study has no decoded MRI slot')
    RUN_RECORD['decode'].append({'tag': tag, 'studies': len(studies),
        'cache_bytes': cache.nbytes, 'filled_slots': int(mask.sum()),
        'failed_series': len(DECODE_FAILED)})
    return studies, cache, mask
'''

RUNTIME = '''
import importlib.metadata
import platform

ROOT = single_directory([p for p in (
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/rsna-knee-abnormality-detection'))
    if (p / 'test.csv').is_file() and (p / 'test_series').is_dir()], 'competition directory')
if not torch.cuda.is_available():
    raise RuntimeError('This reference requires the configured free Kaggle GPU')
if Path('submission.csv').exists():
    raise RuntimeError('Refusing to reuse a pre-existing submission.csv')
ORDER_CACHE = None
RUN_RECORD = {'source_sha256': EXPECTED_SOURCE_SHA256,
              'manifest_sha256': EXPECTED_MANIFEST_SHA256,
              'members': [], 'decode': [], 'seed': SEED,
              'device': torch.cuda.get_device_name(0), 'python': platform.python_version(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ('torch', 'transformers', 'numpy', 'pandas', 'pydicom')}}
try:
    package = checked_package()
    RUN_RECORD['model_path'] = str(find_dinov2())
    sub = infer_from_package(package, torch.device('cuda'))
    RUN_RECORD['submission_sha256'] = file_sha256('submission.csv')
    RUN_RECORD['rows'] = len(sub)
    RUN_RECORD['status'] = 'complete'
except Exception as exc:
    RUN_RECORD['status'] = 'failed'
    RUN_RECORD['error'] = f'{type(exc).__name__}: {exc}'
    raise
finally:
    RUN_RECORD['elapsed_seconds'] = time.time() - T0
    Path('run_manifest.json').write_text(json.dumps(RUN_RECORD, indent=2) + '\\n')
'''


def build_reference(source_path: Path, manifest_path: Path, output: Path):
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Notebook source differs from the reviewed snapshot')
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Weights manifest differs from the reviewed snapshot')
    source = json.loads(source_path.read_text())
    # Only model/pixel/inference definitions. Never copy report extraction or main().
    indices = (13, 15, 16, 19, 21, 23, 25, 26, 28, 30)
    code = '\n\n'.join(''.join(source['cells'][i]['source']) for i in indices)
    code = checked_replace(code, 'bb = AutoModel.from_pretrained(str(p))',
                           'bb = AutoModel.from_pretrained(str(p), local_files_only=True, trust_remote_code=False)')
    code = checked_replace(code, 'weights_only=False', 'weights_only=True')
    code = checked_replace(code, 'ck["fingerprint"]', 'm["fingerprint"]')
    code = checked_replace(code, '    if d > tol:',
                           '    if not np.isfinite(got).all() or not np.isfinite(exp).all() or d > tol:')
    code = checked_replace(code, '    if any(k is None for k, _ in keyed):',
                           '    if any(k is None or not np.isfinite(k) for k, _ in keyed):')
    code = checked_replace(code, '        return files, False',
                           '        raise RuntimeError("Series lacks reliable slice geometry/order")')
    code = checked_replace(code, '    for f in files:\n        k = None',
                           '    for f in files:\n        ds = None\n        k = None')
    code = checked_replace(code, '                rec["ordered"] = files\n                ok += int(good)',
                           '                if not good:\n                    raise RuntimeError("Slice ordering failed")\n                rec["ordered"] = files\n                ok += int(good)')
    code = checked_replace(code, '                break\n    if ORDER_CACHE and done:',
                           '                raise RuntimeError("Ordering time budget exhausted")\n    if ORDER_CACHE and done:')
    code = checked_replace(code, '                break\n    n_failed = len(DECODE_FAILED)',
                           '                raise RuntimeError("Decoding time budget exhausted")\n    n_failed = len(DECODE_FAILED)')
    start = code.index('            if fixed_s is not None and per_win_s is not None:')
    end = code.index('            t0 = time.time()', start)
    code = code[:start] + ('            if left <= 0:\n'
                         '                raise RuntimeError("Full ensemble time budget exhausted")\n') + code[end:]
    code = checked_replace(code, '            ck = torch.load(Path(path) / m["file"], map_location="cpu",',
                           '            checkpoint_path = Path(path) / m["file"]\n'
                           '            checkpoint_hash = file_sha256(checkpoint_path)\n'
                           '            ck = torch.load(checkpoint_path, map_location="cpu",')
    code = checked_replace(code, '            per_member.append({"id": m["id"], "ids": st_te, "pred": p,',
                           '            if p.shape != (len(idx), 12) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():\n'
                           '                raise ValueError("Invalid member probabilities")\n'
                           '            RUN_RECORD["members"].append({"id": m["id"], "sha256": checkpoint_hash, "windows": len(starts)})\n'
                           '            per_member.append({"id": m["id"], "ids": st_te, "pred": p,')
    # Remove the unused permissive package scanner; runtime requires the exact manifest.
    tree = ast.parse(code)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_weights')
    lines = code.splitlines(keepends=True)
    code = ''.join(lines[:node.lineno - 1] + lines[node.end_lineno:])
    code += '\n\n' + inspect.getsource(strict_submission) + RUNTIME_HELPERS
    preamble = ('import os\nos.environ["HF_HUB_OFFLINE"] = "1"\n'
                'os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
                f'EXPECTED_SOURCE_SHA256 = {SOURCE_SHA256!r}\n'
                f'EXPECTED_MANIFEST_SHA256 = {MANIFEST_SHA256!r}\n')
    cells = []
    for kind, text in [('markdown', '# RSNA Knee Image Reference\n\n'
                       'Adapted from pilkwang/rsna-knee-baseline-v1 (Apache 2.0, Kaggle public source). '
                       'Private offline inference using public CC0 weights and DINOv2-small. '
                       'External trained reference; not independent validation on our folds. '
                       'All 20 members and 10 windows per member must complete.\n'),
                       ('code', preamble), ('code', code), ('code', RUNTIME)]:
        if kind == 'code':
            compile(text, '<generated cell>', 'exec')
        cell = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    metadata = {'id': 'willmurray99/rsna-knee-image-reference', 'title': 'RSNA Knee Image Reference',
                'code_file': 'image-reference.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'competition_sources': ['rsna-knee-abnormality-detection'],
                'dataset_sources': ['pilkwang/rsna-knee-weights'],
                'model_sources': ['metaresearch/dinov2/PyTorch/small/1'], 'kernel_sources': []}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'image-reference.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'build_manifest.json').write_text(json.dumps({
        'source_sha256': SOURCE_SHA256, 'weights_manifest_sha256': MANIFEST_SHA256,
        'notebook_sha256': hashlib.sha256((output / 'image-reference.ipynb').read_bytes()).hexdigest(),
        'source_url': 'https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_reference(args.source, args.manifest, args.output)
    print(args.output)
