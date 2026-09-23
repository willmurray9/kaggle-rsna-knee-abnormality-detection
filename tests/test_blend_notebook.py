"""Exercise the packaged offline artifact and its immutable component boundary."""

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rsnaknee.data import sha256
from test_blend import runnable_components


def builder_module():
    try:
        return importlib.import_module('rsnaknee.blend_notebook')
    except ModuleNotFoundError:
        pytest.fail('The fixed blend notebook builder is missing')


@pytest.fixture
def saved_components(tmp_path, monkeypatch):
    module = builder_module()
    data, expected, notebooks = runnable_components(tmp_path)
    roots, pins = {}, {}
    for role in ('public', 'independent'):
        folder = tmp_path / f'saved-{role}'; folder.mkdir(); (folder / 'output').mkdir()
        roots[role] = folder
        filename = 'image-reference.ipynb' if role == 'public' else 'adapted-image.ipynb'
        (folder / filename).write_bytes(notebooks[role])
        manifest_name = 'run_manifest.json' if role == 'public' else 'inference_manifest.json'
        (folder / 'output' / manifest_name).write_bytes((tmp_path / 'work' / role / manifest_name).read_bytes())
        build = {'notebook_sha256': sha256(folder / filename)}
        if role == 'public':
            build.update(source_sha256='source', weights_manifest_sha256='weights')
        else:
            build.update(expected['independent'], source_sha256={})
        (folder / 'build_manifest.json').write_text(json.dumps(build))
        metadata = {'is_private': True, 'enable_internet': False, 'enable_gpu': True,
                    'competition_sources': ['rsna-knee-abnormality-detection'],
                    'model_sources': ['metaresearch/dinov2/PyTorch/small/1'],
                    'dataset_sources': ['pilkwang/rsna-knee-weights'] if role == 'public' else [],
                    'kernel_sources': [] if role == 'public' else ['willmurray99/rsna-knee-all-window-training']}
        (folder / 'kernel-metadata.json').write_text(json.dumps(metadata))
        pins[role] = {name: sha256(folder / name) for name in (filename, 'build_manifest.json', 'kernel-metadata.json', f'output/{manifest_name}')}
    monkeypatch.setattr(module, 'SOURCE_PINS', pins)
    return data, roots


def test_built_notebook_executes_without_repository_or_training_inputs(tmp_path, saved_components):
    module = builder_module()
    data, roots = saved_components
    output = tmp_path / 'build'
    module.build_blend_notebook(roots['public'], roots['independent'], output)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'willmurray99/rsna-knee-reference-blend'
    assert metadata['is_private'] and not metadata['enable_internet']
    assert metadata['enable_gpu'] and metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['pilkwang/rsna-knee-weights']
    assert metadata['kernel_sources'] == ['willmurray99/rsna-knee-all-window-training']
    assert metadata['model_sources'] == ['metaresearch/dinov2/PyTorch/small/1']
    manifest = json.loads((output / 'build_manifest.json').read_text())
    assert manifest['notebook_sha256'] == sha256(output / metadata['code_file'])
    assert manifest['kernel_metadata_sha256'] == sha256(output / 'kernel-metadata.json')
    assert manifest['builder_sha256'] == sha256(Path(module.__file__))
    notebook = json.loads((output / metadata['code_file']).read_text())
    work = tmp_path / 'execution'; work.mkdir()
    code = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            compile(source, 'built-notebook', 'exec')
            source = source.replace("Path('/kaggle/working')", f'Path({str(work)!r})')
            source = source.replace("Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')", f'Path({str(data)!r})')
            code.append(source)
    runner = tmp_path / 'execute.py'
    runner.write_text(f'cells = {code!r}\nns = {{}}\nfor cell in cells:\n    exec(compile(cell, "cell", "exec"), ns)\n')
    subprocess.run([sys.executable, str(runner)], cwd=work, check=True)
    result = json.loads((work / 'blend_manifest.json').read_text())
    assert result['studies'] == 3
    for role, filename in [('public', 'image-reference.ipynb'), ('independent', 'adapted-image.ipynb')]:
        assert (work / role / 'original.ipynb').read_bytes() == (roots[role] / filename).read_bytes()
    for name, digest in manifest['source_sha256'].items():
        assert sha256(work / 'rsnaknee' / name) == digest
    with pytest.raises(FileExistsError): module.build_blend_notebook(roots['public'], roots['independent'], output)


@pytest.mark.parametrize('role,file', [('public', 'image-reference.ipynb'), ('public', 'output/run_manifest.json'), ('public', 'kernel-metadata.json'), ('independent', 'adapted-image.ipynb'), ('independent', 'build_manifest.json')])
def test_source_or_dependency_edits_cannot_be_packaged(tmp_path, saved_components, role, file):
    module = builder_module()
    _, roots = saved_components
    path = roots[role] / file
    path.write_bytes(path.read_bytes() + b' ')
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='hash'): module.build_blend_notebook(roots['public'], roots['independent'], output)
    assert not output.exists()


def test_actual_saved_notebooks_preserve_every_embedded_source_and_pin(tmp_path):
    module = builder_module()
    public = Path('artifacts/kaggle/image-reference/versions/v1')
    independent = Path('artifacts/kaggle/all-window-image/versions/v1')
    if not public.exists() or not independent.exists():
        pytest.skip('Saved private notebook artifacts are intentionally not committed')
    module.build_blend_notebook(public, independent, tmp_path / 'build')
    manifest = json.loads((tmp_path / 'build/build_manifest.json').read_text())
    assert len(manifest['expected_provenance']['public']['members']) == 20
    assert manifest['expected_provenance']['independent']['model_sha256'] == 'd1a796a6945837c7a503271bfb31d028671d41caa340c89349a8972082861172'
    from rsnaknee.blend import prepare_component
    import ast
    original = (independent / 'adapted-image.ipynb').read_bytes()
    runner = prepare_component(tmp_path / 'child', original, hashlib.sha256(original).hexdigest(), 'independent')
    assignments = {node.targets[0].id: node.value for node in ast.parse(runner.read_text()).body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    cells = ast.literal_eval(assignments['cells'])
    old_cells = [''.join(cell['source']) for cell in json.loads(original)['cells'] if cell['cell_type'] == 'code']
    def embedded_sources(cell):
        return next(ast.literal_eval(node.value) for node in ast.parse(cell).body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'sources' for target in node.targets))
    assert embedded_sources(cells[0]) == embedded_sources(old_cells[0])
    assert len(embedded_sources(cells[0])) == 12
    assert cells[1:] == old_cells[1:]
