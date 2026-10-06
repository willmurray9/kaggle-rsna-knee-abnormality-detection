"""Package a pinned public notebook, unchanged, for private offline Kaggle inference."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PINS = {
    'public-frontier': {
        'source_kernel': 'yamadan96/rsna-knee-d4-public0946', 'source_version': 2,
        'source_sha256': '35547f8a28ed09fc68acefa6d23f12fa0bfd6a959f54fdadfdfc3dc3c32768f8',
        'source_metadata_sha256': 'a69a39714c6e33686aa2b784c75023377689dc1c0d0d9a86376e1167840bca55',
        'kernel_id': 'willmurray99/rsna-knee-public-frontier', 'title': 'RSNA Knee Public Frontier',
        'code_file': 'public-frontier.ipynb'},
    'stack-reader': {
        'source_kernel': 'goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944', 'source_version': 5,
        'source_sha256': '941802c45fbc21a763c259d54d35f5fe1600f3a269a39d6773edcbf4185f457e',
        'source_metadata_sha256': 'ad567d8ff5ea7bf253264fcb2426c9314a3fd9e5435017b87873abf642e6db18',
        'kernel_id': 'willmurray99/rsna-knee-stack-reader', 'title': 'RSNA Knee Stack Reader',
        'code_file': 'stack-reader.ipynb'},
}
SOURCE_FIELDS = ('competition_sources', 'dataset_sources', 'kernel_sources', 'model_sources')


def git_state():
    run = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout.strip()
    return {'revision': run('rev-parse', 'HEAD'),
            'tracked_changes': bool(run('status', '--porcelain', '--untracked-files=no'))}


def build_frontier(source_path: Path, source_metadata_path: Path, output: Path, candidate='public-frontier'):
    """Copy the reviewed notebook byte-for-byte; only visibility and the kernel id change."""
    pin = PINS[candidate]
    if output.exists():
        raise FileExistsError('Preserve earlier notebook builds')
    notebook = source_path.read_bytes()
    if hashlib.sha256(notebook).hexdigest() != pin['source_sha256']:
        raise ValueError('Notebook source differs from the reviewed snapshot')
    if hashlib.sha256(source_metadata_path.read_bytes()).hexdigest() != pin['source_metadata_sha256']:
        raise ValueError('Kernel metadata differs from the reviewed snapshot')
    source = json.loads(source_metadata_path.read_text())
    if source['id'] != pin['source_kernel'] or source['enable_internet'] or not source['enable_gpu']:
        raise ValueError('Unexpected source kernel settings')
    sources = {field: list(source.get(field, [])) for field in SOURCE_FIELDS}
    if not all(sources['dataset_sources']) or sources['competition_sources'] != ['rsna-knee-abnormality-detection']:
        raise ValueError('Every attached source must be public and resolvable')
    metadata = {'id': pin['kernel_id'], 'title': pin['title'], 'code_file': pin['code_file'],
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': False,
                'machine_shape': source['machine_shape'], 'docker_image': source['docker_image'],
                **sources}
    output.mkdir(parents=True)
    (output / pin['code_file']).write_bytes(notebook)
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'build_manifest.json').write_text(json.dumps({
        'candidate': candidate, 'source_kernel': pin['source_kernel'], 'source_version': pin['source_version'],
        'source_url': f"https://www.kaggle.com/code/{pin['source_kernel']}",
        'source_sha256': pin['source_sha256'], 'source_metadata_sha256': pin['source_metadata_sha256'],
        'notebook_sha256': hashlib.sha256(notebook).hexdigest(), 'git': git_state()}, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=sorted(PINS), default='public-frontier')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_frontier(args.source, args.metadata, args.output, args.candidate)
    print(args.output)
