import hashlib
import json
from pathlib import Path

import pytest

from rsnaknee import frontier

SOURCE = Path('artifacts/research/public-frontier/rsna-knee-d4-public0946.ipynb')
METADATA = Path('artifacts/research/public-frontier/kernel-metadata.json')


def write_source(tmp_path, monkeypatch, **overrides):
    notebook = tmp_path / 'source.ipynb'
    notebook.write_text('{"cells": []}')
    meta = {'id': frontier.SOURCE_KERNEL, 'enable_gpu': True, 'enable_internet': False,
            'machine_shape': 'NvidiaTeslaT4', 'docker_image': 'gcr.io/x@sha256:1',
            'competition_sources': ['rsna-knee-abnormality-detection'],
            'dataset_sources': ['a/b'], 'kernel_sources': ['c/d'], 'model_sources': ['e/f/g/1'],
            **overrides}
    metadata = tmp_path / 'kernel-metadata.json'
    metadata.write_text(json.dumps(meta))
    monkeypatch.setattr(frontier, 'SOURCE_SHA256', hashlib.sha256(notebook.read_bytes()).hexdigest())
    monkeypatch.setattr(frontier, 'SOURCE_METADATA_SHA256', hashlib.sha256(metadata.read_bytes()).hexdigest())
    return notebook, metadata


def test_copy_is_private_offline_and_keeps_every_source(tmp_path, monkeypatch):
    notebook, metadata = write_source(tmp_path, monkeypatch)
    meta = frontier.build_frontier(notebook, metadata, tmp_path / 'out')
    assert (tmp_path / 'out' / frontier.CODE_FILE).read_bytes() == notebook.read_bytes()
    assert meta['id'] == frontier.KERNEL_ID and meta['is_private'] is True
    assert meta['enable_internet'] is False and meta['enable_gpu'] is True
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['docker_image'] == 'gcr.io/x@sha256:1'
    assert meta['dataset_sources'] == ['a/b'] and meta['kernel_sources'] == ['c/d']
    assert meta['model_sources'] == ['e/f/g/1']
    with pytest.raises(FileExistsError):
        frontier.build_frontier(notebook, metadata, tmp_path / 'out')


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


def test_pinned_snapshot_builds_when_downloaded(tmp_path):
    if not SOURCE.exists() or not METADATA.exists():
        pytest.skip('Downloaded public source is intentionally not committed')
    meta = frontier.build_frontier(SOURCE, METADATA, tmp_path / 'out')
    assert len(meta['dataset_sources']) == 14 and len(meta['kernel_sources']) == 2
