"""Package the two scored, immutable notebooks as one private offline rank blend."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import tempfile

from rsnaknee.blend import prepare_component
from rsnaknee.data import sha256


SOURCE_PINS = {
    'public': {
        'image-reference.ipynb': 'de1130fdac3ef546ef30ed7d1db3826e78c11356267dbc8519196cf2f7f1a55f',
        'build_manifest.json': '17259893cce7811506789b9800dd391c8e58eee19eb361055f5449a3edf5a07f',
        'kernel-metadata.json': 'fb312571b534970f0d0408e74c248c755f5e7f82eb180e295411e04f17207d81',
        'output/run_manifest.json': 'e532d03d67fe592c5e200032a0302777d4d085059b886d5b87d88aae4be050b1',
    },
    'independent': {
        'adapted-image.ipynb': '3b4ce9dd29b6f5c0bfe8d4e7f5885c979efd1a2d4ba43d7a33a81e9f65150100',
        'build_manifest.json': '864c61685669614a23a6f8d63cc3916fc99376479811e1d8eccba79d5fc72528',
        'kernel-metadata.json': 'add32c48489acf484e8e1b3d96717a5dc87b7bd7d1fdd308f58e750abe03accc',
        'output/inference_manifest.json': '3b42ebb7dc0d086f644b86787b6dd6dcfcbed3b47c243e88514be13eeb6e2d4b',
    },
}
SOURCES = ('__init__.py', 'constants.py', 'submission.py', 'data.py', 'blend.py')


def build_blend_notebook(public_source: Path, independent_source: Path, output: Path):
    if output.exists():
        raise FileExistsError('Preserve earlier notebook builds')
    roots = {'public': public_source, 'independent': independent_source}
    notebooks, manifests, builds = {}, {}, {}
    runner_hashes = {}
    for role, root in roots.items():
        for filename, digest in SOURCE_PINS[role].items():
            if sha256(root / filename) != digest:
                raise ValueError(f'Pinned {role} source hash differs: {filename}')
        filename = 'image-reference.ipynb' if role == 'public' else 'adapted-image.ipynb'
        notebooks[role] = (root / filename).read_bytes()
        builds[role] = json.loads((root / 'build_manifest.json').read_text())
        name = 'run_manifest.json' if role == 'public' else 'inference_manifest.json'
        manifests[role] = json.loads((root / 'output' / name).read_text())
        if manifests[role].get('status') != 'complete':
            raise ValueError('Require completed scored components')
        if builds[role]['notebook_sha256'] != sha256(root / filename):
            raise ValueError('Component build notebook hash differs')
        with tempfile.TemporaryDirectory() as temp:
            runner = prepare_component(Path(temp) / role, notebooks[role], builds[role]['notebook_sha256'], role)
            runner_hashes[role] = sha256(runner)
    public = manifests['public']
    independent = manifests['independent']
    if (len(public['members']) != 20 or len({member['id'] for member in public['members']}) != 20
            or any(member['windows'] != 10 for member in public['members'])):
        raise ValueError('Require all 20 public members with all ten windows')
    expected = {
        'public': {key: public[key] for key in ('source_sha256', 'manifest_sha256', 'members')},
        'independent': {key: independent[key] for key in ('arm', 'train_windows_per_plane',
            'training_summary_sha256', 'model_sha256', 'window_schedule_sha256')},
        'source_notebook_sha256': {role: builds[role]['notebook_sha256'] for role in roots},
    }
    if (independent['arm'] != 'all_windows' or independent['train_windows_per_plane'] != 10):
        raise ValueError('Require the scored all-ten-window independent model')
    sources = {name: Path(__file__).with_name(name).read_text() for name in SOURCES}
    source_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    payload = {role: base64.b64encode(content).decode() for role, content in notebooks.items()}
    bootstrap = (
        'import base64, hashlib, json, sys\nfrom pathlib import Path\n'
        "working = Path('/kaggle/working')\n"
        "package = working / 'rsnaknee'\npackage.mkdir(exist_ok=True)\n"
        f'sources = {sources!r}\nsource_sha256 = {source_hashes!r}\n'
        'for name, source in sources.items():\n'
        '    if hashlib.sha256(source.encode()).hexdigest() != source_sha256[name]:\n'
        '        raise ValueError("Blend runtime source hash differs")\n'
        '    (package / name).write_text(source)\n'
        'sys.path.insert(0, str(working))\n'
        f'payload = {payload!r}\nEXPECTED = {expected!r}\n'
        'notebooks = {role: base64.b64decode(encoded) for role, encoded in payload.items()}\n'
    )
    runtime = '''from rsnaknee.blend import run_blend
roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
         Path('/kaggle/input/rsna-knee-abnormality-detection')]
roots = [root for root in roots if (root / 'test.csv').is_file()]
if len(roots) != 1:
    raise ValueError('Require exactly one competition data mount')
result = run_blend(roots[0], working, notebooks, EXPECTED)
print(json.dumps(result, indent=2))
'''
    cells = []
    for kind, source, role in (
        ('markdown', '# Fixed public-reference blend\n\n90% public ensemble ranks and 10% independent '
         'all-ten-window model ranks. Public component: [pilkwang baseline]'
         '(https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) and '
         '[weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights). '
         'Average ties, per target across the full runtime test set. '
         'No training, labels, blend search or partial-ensemble fallback. Original scored notebooks '
         'run in sequential processes under a shared eight-hour limit.\n', 'description'),
        ('code', bootstrap, 'bootstrap'), ('code', runtime, 'runtime'),
    ):
        cell = {'cell_type': kind, 'metadata': {'role': role}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            compile(source, 'reference-blend', 'exec')
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}}
    metadata = {'id': 'willmurray99/rsna-knee-reference-blend', 'title': 'RSNA Knee Reference Blend',
                'code_file': 'reference-blend.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': False,
                'machine_shape': 'NvidiaTeslaT4', 'competition_sources': ['rsna-knee-abnormality-detection'],
                'dataset_sources': ['pilkwang/rsna-knee-weights'],
                'model_sources': ['metaresearch/dinov2/PyTorch/small/1'],
                'kernel_sources': ['willmurray99/rsna-knee-all-window-training']}
    output.mkdir(parents=True)
    (output / metadata['code_file']).write_text(json.dumps(notebook, indent=2) + '\n')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    provenance = {'expected_provenance': expected, 'source_input_sha256': SOURCE_PINS,
                  'source_notebook_sha256': expected['source_notebook_sha256'],
                  'source_sha256': source_hashes, 'component_runner_sha256': runner_hashes,
                  'notebook_sha256': sha256(output / metadata['code_file']),
                  'kernel_metadata_sha256': sha256(output / 'kernel-metadata.json'),
                  'builder_sha256': sha256(Path(__file__)), 'execution': 'Build only; no upload or execution performed.'}
    (output / 'build_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-source', type=Path, required=True)
    parser.add_argument('--independent-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_blend_notebook(args.public_source, args.independent_source, args.output)
    print(args.output)
