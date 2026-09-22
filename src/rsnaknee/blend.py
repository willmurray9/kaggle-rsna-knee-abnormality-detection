"""One fixed, test-only blend of the saved public and independent predictions."""

import json

import pandas as pd

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256
from rsnaknee.submission import validate_submission


def test_template(test):
    if list(test.columns) != [ID_COLUMN]:
        raise ValueError('Require runtime test IDs only')
    sample = test.copy()
    sample[TARGET_COLUMNS] = .5
    validate_submission(sample, sample)
    return sample


def align_component(test, frame):
    sample = test_template(test)
    validate_submission(frame, frame)
    if set(frame[ID_COLUMN]) != set(test[ID_COLUMN]):
        raise ValueError('Component IDs differ from runtime test IDs')
    aligned = frame.set_index(ID_COLUMN).loc[test[ID_COLUMN]].reset_index()
    validate_submission(aligned, sample)
    return aligned


def fixed_rank_blend(test, public, independent):
    """Rank each target over the entire test set; average ties; fixed 90/10 weights."""
    public = align_component(test, public)
    independent = align_component(test, independent)
    result = test.copy().reset_index(drop=True)
    result[TARGET_COLUMNS] = (
        .9 * public[TARGET_COLUMNS].rank(method='average', pct=True)
        + .1 * independent[TARGET_COLUMNS].rank(method='average', pct=True)
    )
    validate_submission(result, test_template(test))
    return result


def verify_component(data_root, working, role, expected):
    name = 'run_manifest.json' if role == 'public' else 'inference_manifest.json'
    path = working / role
    manifest = json.loads((path / name).read_text())
    test = pd.read_csv(data_root / 'test.csv', dtype=str)
    if manifest.get('status') != 'complete':
        raise ValueError(f'{role} did not complete')
    for key, value in expected[role].items():
        if key == 'members':
            members = manifest.get(key, [])
            if (len(members) != 20 or len({m['id'] for m in members}) != 20
                    or sorted(members, key=lambda m: m['id']) != sorted(value, key=lambda m: m['id'])):
                raise ValueError('Public ensemble checkpoint hashes/windows/member count differ')
        elif manifest.get(key) != value:
            raise ValueError(f'{role} provenance differs: {key}')
    count = 'rows' if role == 'public' else 'studies'
    if manifest.get(count) != len(test):
        raise ValueError(f'{role} study count differs')
    if role == 'independent':
        for file in ('test.csv', 'test_series.csv'):
            if manifest.get(file.replace('.csv', '_sha256')) != sha256(data_root / file):
                raise ValueError('Independent test metadata hash differs')
    csv_path = path / 'submission.csv'
    if manifest.get('submission_sha256') != sha256(csv_path):
        raise ValueError(f'{role} submission hash differs')
    predictions = align_component(test, pd.read_csv(csv_path, dtype={ID_COLUMN: str}))
    evidence = {'manifest_sha256': sha256(path / name), 'submission_sha256': sha256(csv_path),
                'provenance': {key: manifest[key] for key in expected[role]}}
    return predictions, evidence


def finalize_blend(data_root, working, expected, component_seconds):
    """Fail closed before writing the candidate if either component is invalid."""
    output = working / 'submission.csv'
    if output.exists() or (working / 'blend_manifest.json').exists():
        raise FileExistsError('Refusing to reuse a previous blend')
    public, public_evidence = verify_component(data_root, working, 'public', expected)
    independent, independent_evidence = verify_component(data_root, working, 'independent', expected)
    test = pd.read_csv(data_root / 'test.csv', dtype=str)
    result = fixed_rank_blend(test, public, independent)
    with output.open('x') as handle:
        result.to_csv(handle, index=False)
    manifest = {
        'status': 'complete', 'studies': len(result),
        'recipe': {'public': .9, 'independent': .1, 'rank': 'average ties / total study count',
                   'scope': 'each target across the full dynamic test set'},
        'source_notebook_sha256': expected['source_notebook_sha256'],
        'components': {'public': public_evidence, 'independent': independent_evidence},
        'component_seconds': component_seconds,
        'test_sha256': sha256(data_root / 'test.csv'),
        'test_series_sha256': sha256(data_root / 'test_series.csv'),
        'submission_sha256': sha256(output), 'labels_or_reports_read': False,
    }
    with (working / 'blend_manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    return manifest


WALL_BUDGET_SECONDS = 8 * 60 * 60


def prepare_component(directory, notebook_bytes, expected_sha256, role):
    """Preserve the scored notebook; execute its cells in a separate process."""
    import hashlib

    if hashlib.sha256(notebook_bytes).hexdigest() != expected_sha256:
        raise ValueError(f'{role} notebook hash differs')
    notebook = json.loads(notebook_bytes)
    cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    if not cells:
        raise ValueError('Component notebook has no code')
    if role == 'independent':
        declaration = "working = Path('/kaggle/working')"
        if sum(cell.count(declaration) for cell in cells) != 1:
            raise ValueError('Require exactly one independent working-directory assignment')
        cells = [cell.replace(declaration, 'working = Path.cwd()') for cell in cells]
    for index, code in enumerate(cells):
        compile(code, f'{role}-cell-{index}', 'exec')
    directory.mkdir()
    (directory / 'original.ipynb').write_bytes(notebook_bytes)
    runner = directory / 'runner.py'
    runner.write_text(
        'import os\nos.environ["HF_HUB_OFFLINE"] = "1"\n'
        'os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
        f'cells = {cells!r}\nnamespace = {{"__name__": "__main__"}}\n'
        'for index, code in enumerate(cells):\n'
        '    exec(compile(code, f"component-cell-{index}", "exec"), namespace)\n'
    )
    return runner


def run_blend(data_root, working, notebooks, expected):
    """Run the unchanged components sequentially within one shared eight-hour budget."""
    import subprocess
    import sys
    import time

    started = time.perf_counter()
    working = working.resolve()
    if (working / 'submission.csv').exists() or (working / 'blend_manifest.json').exists():
        raise FileExistsError('Refusing to reuse a previous blend')
    test_template(pd.read_csv(data_root / 'test.csv', dtype=str))
    input_hashes = {name: sha256(data_root / name) for name in ('test.csv', 'test_series.csv')}
    component_seconds = {}
    for role in ('public', 'independent'):
        directory = working / role
        runner = prepare_component(directory, notebooks[role], expected['source_notebook_sha256'][role], role)
        remaining = WALL_BUDGET_SECONDS - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError('Shared blend wall budget exhausted')
        print(f'Starting {role}; shared budget remaining {remaining:.0f}s', flush=True)
        component_started = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(runner)], cwd=directory, check=True, timeout=remaining)
        component_seconds[role] = time.perf_counter() - component_started
        if any(sha256(data_root / name) != digest for name, digest in input_hashes.items()):
            raise ValueError('Runtime test metadata changed during component execution')
        verify_component(data_root, working, role, expected)
    if time.perf_counter() - started >= WALL_BUDGET_SECONDS:
        raise TimeoutError('Shared blend wall budget exhausted')
    result = finalize_blend(data_root, working, expected, component_seconds)
    result['elapsed_seconds'] = time.perf_counter() - started
    result['wall_budget_seconds'] = WALL_BUDGET_SECONDS
    result['component_runner_sha256'] = {role: sha256(working / role / 'runner.py') for role in ('public', 'independent')}
    (working / 'blend_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
