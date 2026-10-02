"""Package the pinned public 0.943 frontier notebook, unchanged, for private offline inference."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_KERNEL = 'yamadan96/rsna-knee-d4-public0946'
SOURCE_VERSION = 2
SOURCE_SHA256 = '35547f8a28ed09fc68acefa6d23f12fa0bfd6a959f54fdadfdfc3dc3c32768f8'
SOURCE_METADATA_SHA256 = 'a69a39714c6e33686aa2b784c75023377689dc1c0d0d9a86376e1167840bca55'
KERNEL_ID = 'willmurray99/rsna-knee-public-frontier'
CODE_FILE = 'public-frontier.ipynb'
SOURCE_FIELDS = ('competition_sources', 'dataset_sources', 'kernel_sources', 'model_sources')


def build_frontier(source_path: Path, source_metadata_path: Path, output: Path):
    """Copy the reviewed notebook byte-for-byte; only visibility and the kernel id change."""
    if output.exists():
        raise FileExistsError('Preserve earlier notebook builds')
    notebook = source_path.read_bytes()
    if hashlib.sha256(notebook).hexdigest() != SOURCE_SHA256:
        raise ValueError('Notebook source differs from the reviewed snapshot')
    if hashlib.sha256(source_metadata_path.read_bytes()).hexdigest() != SOURCE_METADATA_SHA256:
        raise ValueError('Kernel metadata differs from the reviewed snapshot')
    source = json.loads(source_metadata_path.read_text())
    if source['id'] != SOURCE_KERNEL or source['enable_internet'] or not source['enable_gpu']:
        raise ValueError('Unexpected source kernel settings')
    sources = {field: list(source.get(field, [])) for field in SOURCE_FIELDS}
    if not all(sources['dataset_sources']) or sources['competition_sources'] != ['rsna-knee-abnormality-detection']:
        raise ValueError('Every attached source must be public and resolvable')
    metadata = {'id': KERNEL_ID, 'title': 'RSNA Knee Public Frontier', 'code_file': CODE_FILE,
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': False,
                'machine_shape': source['machine_shape'], 'docker_image': source['docker_image'],
                **sources}
    output.mkdir(parents=True)
    (output / CODE_FILE).write_bytes(notebook)
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'build_manifest.json').write_text(json.dumps({
        'source_kernel': SOURCE_KERNEL, 'source_version': SOURCE_VERSION,
        'source_url': f'https://www.kaggle.com/code/{SOURCE_KERNEL}',
        'source_sha256': SOURCE_SHA256, 'source_metadata_sha256': SOURCE_METADATA_SHA256,
        'notebook_sha256': hashlib.sha256(notebook).hexdigest()}, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_frontier(args.source, args.metadata, args.output)
    print(args.output)
