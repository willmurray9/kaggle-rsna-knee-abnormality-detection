"""Catch wrong ranking/alignment, incomplete branches and stale output reuse."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from rsnaknee.constants import ID_COLUMN, TARGET_COLUMNS
from rsnaknee.data import sha256


def blend_module():
    try:
        return importlib.import_module('rsnaknee.blend')
    except ModuleNotFoundError:
        pytest.fail('The fixed rank blend implementation is missing')


def predictions(ids, values):
    frame = pd.DataFrame({ID_COLUMN: ids})
    frame[TARGET_COLUMNS] = np.repeat(np.asarray(values)[:, None], 12, axis=1)
    return frame


def test_fixed_blend_uses_average_ties_and_aligns_ids_before_global_ranking():
    module = blend_module()
    test = pd.DataFrame({ID_COLUMN: ['c', 'a', 'b', 'd']})
    public = predictions(['a', 'b', 'c', 'd'], [.1, .1, .6, .9])
    own = predictions(['d', 'b', 'a', 'c'], [.1, .4, .9, .2])
    result = module.fixed_rank_blend(test, public, own)
    assert result[ID_COLUMN].tolist() == ['c', 'a', 'b', 'd']
    # Public ranks: c=.75, a=b=.375, d=1; own: c=.5, a=1, b=.75, d=.25.
    np.testing.assert_allclose(result[TARGET_COLUMNS], np.repeat([[.725], [.4375], [.4125], [.925]], 12, axis=1))


def test_ranks_span_all_1300_replacement_ids_and_target_columns():
    module = blend_module()
    n = 1300
    ids = [f'hidden-{i:04}' for i in range(n)]
    test = pd.DataFrame({ID_COLUMN: ids})
    public = predictions(ids, np.arange(n) / n)
    own = predictions(ids, np.arange(n)[::-1] / n)
    public[TARGET_COLUMNS[1]] = 1 - public[TARGET_COLUMNS[0]]
    result = module.fixed_rank_blend(test, public.sample(frac=1, random_state=1), own.iloc[::-1])
    assert len(result) == n
    for row in (0, 7, 8, 31, 32, 1299):
        assert result.iloc[row, 1] == pytest.approx((.9 * (row + 1) + .1 * (n - row)) / n)
        assert result.iloc[row, 2] == pytest.approx((n - row) / n)


@pytest.mark.parametrize('problem', ['missing', 'extra', 'duplicate', 'blank', 'null', 'target', 'reordered', 'nan', 'inf', 'negative', 'above_one', 'text', 'empty_test'])
def test_invalid_prediction_or_id_never_becomes_a_blend(problem):
    module = blend_module()
    test = pd.DataFrame({ID_COLUMN: ['a', 'b', 'c']})
    public = predictions(['a', 'b', 'c'], [.1, .4, .8])
    own = public.copy()
    if problem == 'missing': public = public.iloc[:2]
    elif problem == 'extra': public.loc[3] = ['d', *([.2] * 12)]
    elif problem == 'duplicate': public.loc[1, ID_COLUMN] = 'a'
    elif problem == 'blank': public.loc[0, ID_COLUMN] = ' '
    elif problem == 'null': public.loc[0, ID_COLUMN] = None
    elif problem == 'target': public = public.drop(columns=[TARGET_COLUMNS[0]])
    elif problem == 'reordered': public = public[[ID_COLUMN, *TARGET_COLUMNS[::-1]]]
    elif problem == 'empty_test': test = test.iloc[:0]
    else:
        values = {'nan': np.nan, 'inf': np.inf, 'negative': -.01, 'above_one': 1.01, 'text': 'invalid'}
        if problem == 'text': public[TARGET_COLUMNS[0]] = public[TARGET_COLUMNS[0]].astype(object)
        public.loc[0, TARGET_COLUMNS[0]] = values[problem]
    with pytest.raises(ValueError): module.fixed_rank_blend(test, public, own)


def component_fixture(tmp_path):
    data = tmp_path / 'input'; data.mkdir()
    work = tmp_path / 'work'; work.mkdir()
    pd.DataFrame({ID_COLUMN: ['c', 'a', 'b']}).to_csv(data / 'test.csv', index=False)
    pd.DataFrame({ID_COLUMN: ['a', 'b', 'c'], 'SeriesInstanceUID': ['1', '2', '3']}).to_csv(data / 'test_series.csv', index=False)
    expected = {
        'public': {'source_sha256': 'source', 'manifest_sha256': 'weights',
                   'members': [{'id': f'm{i}', 'sha256': f'hash{i}', 'windows': 10} for i in range(20)]},
        'independent': {'arm': 'all_windows', 'train_windows_per_plane': 10,
                        'training_summary_sha256': 'summary', 'model_sha256': 'model',
                        'window_schedule_sha256': 'schedule'},
        'source_notebook_sha256': {'public': 'publicnote', 'independent': 'ownnote'},
    }
    for role in ('public', 'independent'):
        path = work / role; path.mkdir()
        predictions(['a', 'b', 'c'], [.1, .4, .8]).to_csv(path / 'submission.csv', index=False)
        manifest = dict(expected[role], status='complete', submission_sha256=sha256(path / 'submission.csv'))
        if role == 'public': manifest['rows'] = 3
        else:
            manifest.update(studies=3, test_sha256=sha256(data / 'test.csv'), test_series_sha256=sha256(data / 'test_series.csv'))
        name = 'run_manifest.json' if role == 'public' else 'inference_manifest.json'
        (path / name).write_text(json.dumps(manifest))
    return data, work, expected


@pytest.mark.parametrize('problem', ['partial', 'wrong_checkpoint', 'wrong_windows', 'duplicate_member', 'failed', 'csv_hash', 'own_model', 'own_summary', 'own_schedule', 'own_test', 'own_count'])
def test_component_provenance_failure_prevents_root_submission(tmp_path, problem):
    module = blend_module()
    data, work, expected = component_fixture(tmp_path)
    role = 'independent' if problem.startswith('own_') else 'public'
    name = 'inference_manifest.json' if role == 'independent' else 'run_manifest.json'
    path = work / role / name; manifest = json.loads(path.read_text())
    if problem == 'partial': manifest['members'].pop()
    elif problem == 'wrong_checkpoint': manifest['members'][0]['sha256'] = 'changed'
    elif problem == 'wrong_windows': manifest['members'][0]['windows'] = 9
    elif problem == 'duplicate_member': manifest['members'][1] = manifest['members'][0]
    elif problem == 'failed': manifest['status'] = 'failed'
    elif problem == 'csv_hash': manifest['submission_sha256'] = 'stale'
    else:
        keys = {'own_model': 'model_sha256', 'own_summary': 'training_summary_sha256', 'own_schedule': 'window_schedule_sha256', 'own_test': 'test_sha256', 'own_count': 'studies'}
        manifest[keys[problem]] = 'changed'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): module.finalize_blend(data, work, expected, {'public': 1., 'independent': 1.})
    assert not (work / 'submission.csv').exists()


def test_final_output_binds_both_components_and_dynamic_metadata(tmp_path):
    module = blend_module()
    data, work, expected = component_fixture(tmp_path)
    result = module.finalize_blend(data, work, expected, {'public': 1., 'independent': 2.})
    assert result['status'] == 'complete' and result['studies'] == 3
    assert result['test_sha256'] == sha256(data / 'test.csv')
    assert result['submission_sha256'] == sha256(work / 'submission.csv')
    for role in ('public', 'independent'):
        assert result['components'][role]['submission_sha256'] == sha256(work / role / 'submission.csv')
    out = pd.read_csv(work / 'submission.csv')
    assert out[ID_COLUMN].tolist() == ['c', 'a', 'b']
    np.testing.assert_allclose(out.iloc[:, 1], [1, 1/3, 2/3])
    before = (work / 'submission.csv').read_bytes()
    with pytest.raises(FileExistsError): module.finalize_blend(data, work, expected, {})
    assert (work / 'submission.csv').read_bytes() == before


def notebook_bytes(cells):
    return json.dumps({'cells': [{'cell_type': 'code', 'source': code.splitlines(keepends=True)} for code in cells]}).encode()


def test_child_notebook_executes_future_imports_and_only_relocates_own_working(tmp_path):
    module = blend_module()
    import hashlib
    import subprocess
    import sys
    content = notebook_bytes([
        "from pathlib import Path\nworking = Path('/kaggle/working')\nsources = {'nested.py': \"literal = '/kaggle/working'\\n\"}\n",
        "from __future__ import annotations\n(working / 'nested.py').write_text(sources['nested.py'])\n",
    ])
    child = tmp_path / 'independent'
    runner = module.prepare_component(child, content, hashlib.sha256(content).hexdigest(), 'independent')
    subprocess.run([sys.executable, str(runner)], cwd=child, check=True)
    assert (child / 'nested.py').read_text() == "literal = '/kaggle/working'\n"
    assert (child / 'original.ipynb').read_bytes() == content


@pytest.mark.parametrize('problem', ['hash', 'missing_assignment', 'duplicate_assignment'])
def test_child_source_corruption_is_rejected_before_execution(tmp_path, problem):
    module = blend_module()
    import hashlib
    text = "from pathlib import Path\nworking = Path('/kaggle/working')\n"
    if problem == 'missing_assignment': text = text.replace('working =', 'other =')
    if problem == 'duplicate_assignment': text += "working = Path('/kaggle/working')\n"
    content = notebook_bytes([text])
    digest = 'wrong' if problem == 'hash' else hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError): module.prepare_component(tmp_path / 'independent', content, digest, 'independent')
    assert not (tmp_path / 'independent/runner.py').exists()


def runnable_components(tmp_path):
    import hashlib
    data, fixture, expected = component_fixture(tmp_path)
    notebooks = {}
    for role in ('public', 'independent'):
        folder = fixture / role
        script = "from pathlib import Path\n"
        if role == 'independent':
            script += "working = Path('/kaggle/working')\nassert Path('../public/finished').is_file()\n"
        for file in folder.iterdir():
            script += f"Path({file.name!r}).write_bytes({file.read_bytes()!r})\n"
        script += "Path('finished').write_text('done')\n"
        notebooks[role] = notebook_bytes([script])
        expected['source_notebook_sha256'][role] = hashlib.sha256(notebooks[role]).hexdigest()
    return data, expected, notebooks


def test_runner_executes_components_sequentially_and_publishes_only_after_success(tmp_path):
    module = blend_module()
    data, expected, notebooks = runnable_components(tmp_path)
    working = tmp_path / 'actual'; working.mkdir()
    result = module.run_blend(data, working, notebooks, expected)
    assert result['status'] == 'complete'
    assert (working / 'submission.csv').exists()
    assert result['elapsed_seconds'] >= sum(result['component_seconds'].values())
    assert all(value > 0 for value in result['component_seconds'].values())


@pytest.mark.parametrize('role', ['public', 'independent'])
def test_failed_child_cannot_leave_a_root_candidate(tmp_path, role):
    module = blend_module()
    import hashlib
    import subprocess
    data, expected, notebooks = runnable_components(tmp_path)
    cells = json.loads(notebooks[role])['cells']
    cells[0]['source'].append("\nraise RuntimeError('deliberate component failure')\n")
    notebooks[role] = json.dumps({'cells': cells}).encode()
    expected['source_notebook_sha256'][role] = hashlib.sha256(notebooks[role]).hexdigest()
    working = tmp_path / 'actual'; working.mkdir()
    with pytest.raises(subprocess.CalledProcessError): module.run_blend(data, working, notebooks, expected)
    assert not (working / 'submission.csv').exists()
    assert 'deliberate component failure' in (working / role / 'execution.log').read_text()
    if role == 'public': assert not (working / 'independent').exists()


def test_shared_remaining_budget_stops_first_child_and_does_not_launch_second(tmp_path, monkeypatch):
    module = blend_module()
    import hashlib
    import subprocess
    data, expected, notebooks = runnable_components(tmp_path)
    notebooks['public'] = notebook_bytes(["import time\nprint('before timeout', flush=True)\ntime.sleep(10)\n"])
    expected['source_notebook_sha256']['public'] = hashlib.sha256(notebooks['public']).hexdigest()
    monkeypatch.setattr(module, 'WALL_BUDGET_SECONDS', .5)
    working = tmp_path / 'actual'; working.mkdir()
    with pytest.raises(subprocess.TimeoutExpired): module.run_blend(data, working, notebooks, expected)
    assert not (working / 'submission.csv').exists()
    assert not (working / 'independent').exists()
    assert (working / 'public/execution.log').read_text() == 'before timeout\n'


def test_verbose_children_write_directly_to_preserved_logs_without_notebook_output(tmp_path, capfd):
    module = blend_module()
    import hashlib
    data, expected, notebooks = runnable_components(tmp_path)
    for role in ('public', 'independent'):
        cells = json.loads(notebooks[role])['cells']
        cells[0]['source'].insert(0, "import os\nos.write(1, b'o' * 1048576)\nos.write(2, b'e' * 1048576)\n")
        notebooks[role] = json.dumps({'cells': cells}).encode()
        expected['source_notebook_sha256'][role] = hashlib.sha256(notebooks[role]).hexdigest()
    working = tmp_path / 'actual'; working.mkdir()
    result = module.run_blend(data, working, notebooks, expected)
    output = capfd.readouterr()
    assert result['status'] == 'complete'
    for role in ('public', 'independent'):
        assert (working / role / 'execution.log').read_bytes() == b'o' * 1048576 + b'e' * 1048576
    assert len(output.out) < 1000 and output.err == ''
