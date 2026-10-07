import hashlib
import json
from pathlib import Path

import pytest

from rsnaknee import frontier

DOWNLOADED = {
    'public-frontier': (Path('artifacts/research/public-frontier/rsna-knee-d4-public0946.ipynb'),
                        Path('artifacts/research/public-frontier/kernel-metadata.json'), 14),
    'stack-reader': (Path('artifacts/research/stack-reader/rsna-knee-stack-2-5d-convnext-mil-lb-0-944.ipynb'),
                     Path('artifacts/research/stack-reader/kernel-metadata.json'), 15),
}


def write_source(tmp_path, monkeypatch, candidate='public-frontier', **overrides):
    notebook = tmp_path / 'source.ipynb'
    notebook.write_text('{"cells": []}')
    meta = {'id': frontier.PINS[candidate]['source_kernel'], 'enable_gpu': True, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4', 'docker_image': 'gcr.io/x@sha256:1',
            'competition_sources': ['rsna-knee-abnormality-detection'],
            'dataset_sources': ['a/b'], 'kernel_sources': ['c/d'], 'model_sources': ['e/f/g/1'],
            **overrides}
    metadata = tmp_path / 'kernel-metadata.json'
    metadata.write_text(json.dumps(meta))
    pin = dict(frontier.PINS[candidate],
               source_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
               source_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    monkeypatch.setitem(frontier.PINS, candidate, pin)
    return notebook, metadata


@pytest.mark.parametrize('candidate', sorted(frontier.PINS))
def test_copy_is_private_offline_and_keeps_every_source(tmp_path, monkeypatch, candidate):
    notebook, metadata = write_source(tmp_path, monkeypatch, candidate)
    meta = frontier.build_frontier(notebook, metadata, tmp_path / 'out', candidate)
    pin = frontier.PINS[candidate]
    assert (tmp_path / 'out' / pin['code_file']).read_bytes() == notebook.read_bytes()
    assert meta['id'] == pin['kernel_id'] and meta['is_private'] is True
    assert meta['enable_internet'] is False and meta['enable_gpu'] is True
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['docker_image'] == 'gcr.io/x@sha256:1'
    assert meta['dataset_sources'] == ['a/b'] and meta['kernel_sources'] == ['c/d']
    assert meta['model_sources'] == ['e/f/g/1']
    manifest = json.loads((tmp_path / 'out' / 'build_manifest.json').read_text())
    assert manifest['candidate'] == candidate and len(manifest['git']['revision']) == 40
    with pytest.raises(FileExistsError):
        frontier.build_frontier(notebook, metadata, tmp_path / 'out', candidate)


def test_candidates_never_share_a_private_kernel():
    assert len({pin['kernel_id'] for pin in frontier.PINS.values()}) == len(frontier.PINS)


@pytest.mark.parametrize('override', [{'dataset_sources': ['a/b', '']}, {'enable_internet': True},
                                      {'id': 'someone/else'}])
def test_unresolvable_or_unexpected_sources_are_rejected(tmp_path, monkeypatch, override):
    notebook, metadata = write_source(tmp_path, monkeypatch, **override)
    with pytest.raises(ValueError):
        frontier.build_frontier(notebook, metadata, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_changed_source_is_rejected(tmp_path, monkeypatch):
    notebook, metadata = write_source(tmp_path, monkeypatch)
    notebook.write_text('{"cells": [1]}')
    with pytest.raises(ValueError, match='Notebook source'):
        frontier.build_frontier(notebook, metadata, tmp_path / 'out')


@pytest.mark.parametrize('candidate', sorted(DOWNLOADED))
def test_pinned_snapshot_builds_when_downloaded(tmp_path, candidate):
    source, metadata, datasets = DOWNLOADED[candidate]
    if not source.exists() or not metadata.exists():
        pytest.skip('Downloaded public source is intentionally not committed')
    meta = frontier.build_frontier(source, metadata, tmp_path / 'out', candidate)
    assert len(meta['dataset_sources']) == datasets and len(meta['kernel_sources']) == 2
