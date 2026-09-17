import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from test_features import image_study, fake_extraction
from rsnaknee import coverage
from rsnaknee.features import normalize_rgb
from rsnaknee.image_model import read_image_features


@pytest.mark.parametrize('image_study', [21], indirect=True)
def test_coverage_uses_twelve_geometric_samples_and_ten_neighboring_windows(image_study):
    root, series = image_study
    prepared = coverage.prepare_study(root, 'train', '1.2', series)
    assert prepared['images'].shape == (3, 12, 224, 224)
    assert json.loads(prepared['series'][0]['sample_indices']) == [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    calls = []
    def encode(pixels):
        calls.append(pixels.copy())
        return np.repeat(pixels.mean(axis=(1, 2, 3))[:, None], 768, axis=1)
    features, windows, presence = coverage.encode_studies([prepared], encode, encoder_batch_size=7)
    actual = np.concatenate(calls)
    assert actual.shape == (30, 3, 224, 224)
    assert max(map(len, calls)) <= 7
    for plane in range(3):
        for window in range(10):
            np.testing.assert_array_equal(actual[plane * 10 + window], normalize_rgb(prepared['images'][plane, window:window + 3][None])[0])
    assert windows.shape == (1, 3, 10, 768) and windows.dtype == np.float16
    np.testing.assert_allclose(features[0, :768], actual[:10].mean(axis=(1, 2, 3)).mean(), atol=1e-6)
    np.testing.assert_array_equal(features[:, -3:], presence)


def test_missing_planes_remain_zero_and_no_usable_plane_rejects(image_study):
    root, series = image_study
    prepared = coverage.prepare_study(root, 'train', '1.2', series[:1])
    features, windows, presence = coverage.encode_studies([prepared], lambda x: np.full((len(x), 768), 3., dtype=np.float32))
    np.testing.assert_array_equal(presence, [[1, 0, 0]])
    assert np.all(features[0, :768] == 3) and np.all(features[0, 768:2304] == 0)
    assert np.all(windows[0, 1:] == 0)
    with pytest.raises(ValueError, match='usable'):
        coverage.encode_studies([coverage.prepare_study(root, 'test', 'absent', [])], lambda x: x)


@pytest.mark.parametrize('bad', [np.nan, np.inf, 70000.])
def test_nonfinite_or_float16_overflow_embeddings_reject(bad):
    prepared = {'images': np.ones((3, 12, 2, 2), np.float32), 'presence': np.array([1, 0, 0])}
    with pytest.raises(ValueError, match='embedding'):
        coverage.encode_studies([prepared], lambda x: np.full((len(x), 768), bad))


def test_extraction_saves_ordered_complete_cache_without_labels(fake_extraction, monkeypatch):
    root, checkpoint, output, _ = fake_extraction
    original_loader = sys.modules['transformers'].AutoModel.from_pretrained
    def load(path, **kwargs):
        assert kwargs.pop('weights_only') is True
        return original_loader(path, **kwargs)
    monkeypatch.setattr(sys.modules['transformers'].AutoModel, 'from_pretrained', load)
    def prepare(root, split, study, rows):
        return {'images': np.broadcast_to(np.array([1, 0, 1], np.float32)[:, None, None, None], (3, 12, 224, 224)), 'presence': np.array([1, 0, 1]),
                'quality': {'StudyInstanceUID': study}, 'series': [{'StudyInstanceUID': study}]}
    monkeypatch.setattr(coverage, 'prepare_study', prepare)
    real_read = pd.read_csv
    def guarded_read(path, *args, **kwargs):
        if Path(path).name in {'train.csv', 'test.csv'}:
            assert kwargs.get('usecols') == ['StudyInstanceUID']
        assert 'report' not in Path(path).name
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', guarded_read)
    # Frozen-model loading itself is exercised through the existing synthetic torch boundary.
    manifest = coverage.extract_features(root, checkpoint, output, device='cpu', batch_size=2, workers=1)
    ids, values, _ = read_image_features(output)
    assert ids.StudyInstanceUID.tolist() == ['b', 'a', 'c']
    assert values.shape == (3, 2307) and np.all(values[:, :384] == 2)
    pixels = np.load(output / 'pixels_uint8.npy', mmap_mode='r')
    assert pixels.shape == (3, 3, 12, 224, 224) and pixels.dtype == np.uint8
    assert np.all(pixels[:, [0, 2]] == 255) and np.all(pixels[:, 1] == 0)
    cache = np.load(output / 'window_features.npy')
    assert cache.shape == (3, 3, 10, 768) and cache.dtype == np.float16
    assert np.all(cache[:, [0, 2], :, 384:] == 4)
    np.testing.assert_array_equal(np.load(output / 'presence.npy'), [[1, 0, 1]] * 3)
    assert manifest['status'] == 'complete' and manifest['completed_studies'] == 3
    assert set(manifest['source_sha256']) == {'__init__.py', 'coverage.py', 'features.py', 'imaging.py'}
    for name, digest in manifest['artifact_sha256'].items():
        assert coverage._hash_file(output / name) == digest
    assert manifest['timing']['probe_studies'] == 2
    with pytest.raises(FileExistsError):
        coverage.extract_features(root, checkpoint, output, device='cpu')


def test_extraction_failure_saves_failed_manifest_and_quality(fake_extraction, monkeypatch):
    root, checkpoint, output, _ = fake_extraction
    original_loader = sys.modules['transformers'].AutoModel.from_pretrained
    def load(path, **kwargs):
        assert kwargs.pop('weights_only') is True
        return original_loader(path, **kwargs)
    monkeypatch.setattr(sys.modules['transformers'].AutoModel, 'from_pretrained', load)
    monkeypatch.setattr(coverage, 'prepare_study', lambda *args: {
        'images': np.zeros((3, 12, 2, 2)), 'presence': np.zeros(3),
        'quality': {'usable_planes': 0}, 'series': [{'status': 'missing'}]})
    with pytest.raises(ValueError, match='usable'):
        coverage.extract_features(root, checkpoint, output, device='cpu')
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'failed'
    assert (output / 'quality.csv').is_file()


def test_float32_pooling_precedes_float16_cache_quantization():
    prepared = {'images': np.ones((3, 12, 2, 2), np.float32), 'presence': np.array([1, 0, 0])}
    features, windows, _ = coverage.encode_studies([prepared], lambda x: np.full((len(x), 768), 1.0003, np.float32))
    np.testing.assert_allclose(features[0, :768], 1.0003, atol=1e-7)
    assert np.all(windows[0, 0] == 1.)


def test_repeated_short_stack_samples_keep_order_and_audit_failed_replacements(image_study):
    import pydicom
    root, series = image_study
    path = root / 'train_series' / '1.2' / '1.2.1' / 'a.dcm'
    ds = pydicom.dcmread(path)
    ds.PixelData = b''
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    result = coverage.prepare_study(root, 'train', '1.2', series)
    selected = result['series'][0]
    assert json.loads(selected['sample_indices']) == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    assert selected['decode_failures'] == selected['repeated_failed_samples'] == 10
    assert result['presence'].tolist() == [1, 1, 1]
    assert np.isfinite(result['images']).all()
    np.testing.assert_array_equal(result['images'][0, 1], result['images'][0, 0])
