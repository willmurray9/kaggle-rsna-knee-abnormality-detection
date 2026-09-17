import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsnaknee.reference import build_reference, checked_replace, strict_submission

TARGETS = [str(i) for i in range(12)]


def test_submission_preserves_ids_and_rejects_invalid_predictions(tmp_path):
    test = pd.DataFrame({'StudyInstanceUID': ['b', 'a']})
    pred = np.tile([0.2, 0.8], (12, 1)).T
    output = tmp_path / 'submission.csv'
    result = strict_submission(pred, ['a', 'b'], test, output, TARGETS)
    assert result.StudyInstanceUID.tolist() == ['b', 'a']
    assert result[TARGETS].iloc[0].eq(1).all()
    for bad, ids in [(pred, ['a', 'a']), (pred, ['a', 'c']), (pred[:1], ['a']),
                     (np.full((2, 12), np.nan), ['a', 'b']),
                     (np.full((2, 12), 1.1), ['a', 'b'])]:
        output.unlink(missing_ok=True)
        with pytest.raises(ValueError):
            strict_submission(bad, ids, test, output, TARGETS)
        assert not output.exists()


def test_patch_assumption_fails_closed():
    with pytest.raises(ValueError):
        checked_replace('same same', 'same', 'new')
    with pytest.raises(ValueError):
        checked_replace('other', 'same', 'new')


def test_unreviewed_source_is_rejected(tmp_path):
    source = tmp_path / 'source.ipynb'
    source.write_text('{}')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='source'):
        build_reference(source, manifest, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_reviewed_notebook_has_no_training_or_unrestricted_loading(tmp_path):
    import ast
    source = Path('artifacts/research/public-baseline/rsna-knee-baseline-v1.ipynb')
    manifest = Path('artifacts/research/public-weights/manifest.json')
    if not source.exists() or not manifest.exists():
        pytest.skip('Downloaded audited source is intentionally not committed')
    build_reference(source, manifest, tmp_path)
    notebook = json.loads((tmp_path / 'image-reference.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    assert 'train.csv' not in code
    assert 'read_labels' not in code
    assert 'write_benchmark_submission' not in code
    assert 'main()' not in code
    assert 'weights_only=False' not in code
    loads = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == 'torch.load']
    assert len(loads) == 1
    assert any(k.arg == 'weights_only' and k.value.value is True for k in loads[0].keywords)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<cell>', 'exec')
    metadata = json.loads((tmp_path / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True
    assert metadata['enable_internet'] is False
    assert metadata['dataset_sources'] == ['pilkwang/rsna-knee-weights']
