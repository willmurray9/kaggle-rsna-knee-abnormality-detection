"""Frozen DINOv2 coverage: twelve central slices, ten neighboring RGB windows.

Baseline preprocessing helpers remain unchanged. Each batch holds only a bounded
number of studies; only a bounded set of windows is transferred to the encoder.
Reports and condition columns are never parsed; input files are hashed as bytes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from rsnaknee.features import (PLANES, HEADER_TAGS, _hash_file, _header_hash,
                               _pixel_spacing, checkpoint_provenance, normalize_rgb)
from rsnaknee.imaging import choose_series, decode_grayscale, order_slices, preprocess_slice


def prepare_study(data_root: Path, split: str, study: str, series: list[dict]) -> dict[str, Any]:
    """Read one selected series per plane; report partial failures instead of hiding them."""
    import pydicom

    images = np.zeros((len(PLANES), 12, 224, 224), dtype=np.float32)
    presence = np.zeros(len(PLANES), dtype=np.float32)
    audits = []
    for plane_index, plane in enumerate(PLANES):
        row = choose_series(series, plane)
        audit = {
            "StudyInstanceUID": study, "split": split, "plane": plane,
            "SeriesInstanceUID": row["SeriesInstanceUID"] if row is not None else "",
            "status": "no_matching_series", "n_files": 0, "header_failures": 0,
            "decode_failures": 0, "repeated_failed_samples": 0, "spacing_fallbacks": 0,
            "order_method": "", "order_fallback_reason": "", "sample_indices": "[]",
            "sample_positions": "[]", "sample_pixel_sha256": "[]", "error_types": "",
            "patient_id_sha256": "", "issuer_sha256": "", "patient_key_sha256": "",
            "scanner_sha256": "", "site_sha256": "", "patient_identity_removed": "",
            "deidentification_method_present": False, "rows": 0, "columns": 0,
            "number_of_frames": 0, "pixel_spacing": "[]", "transfer_syntax": "",
            "laterality": "", "image_laterality": "",
        }
        audits.append(audit)
        if row is None:
            continue
        directory = Path(data_root) / f"{split}_series" / study / str(row["SeriesInstanceUID"])
        files = sorted(directory.glob("*.dcm"))
        audit["n_files"] = len(files)
        if not files:
            audit["status"] = "no_dicom_files"
            continue
        headers, readable = [], []
        errors = set()
        for path in files:
            try:
                header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=HEADER_TAGS)
                if str(getattr(header, "StudyInstanceUID", study)) != study or str(
                    getattr(header, "SeriesInstanceUID", row["SeriesInstanceUID"])
                ) != str(row["SeriesInstanceUID"]):
                    raise ValueError("DICOM identifiers do not match selected metadata")
                headers.append(header)
                readable.append(path)
            except Exception as exc:
                audit["header_failures"] += 1
                errors.add(type(exc).__name__)
        try:
            indices, ordering = order_slices(headers)
        except ValueError:
            audit["status"] = "unorderable_series"
            audit["error_types"] = "|".join(sorted(errors | {"OrderingError"}))
            continue
        ordered_headers = [headers[i] for i in indices]
        ordered_files = [readable[i] for i in indices]
        sample = np.rint(np.linspace(0.2 * (len(indices) - 1), 0.8 * (len(indices) - 1), 12)).astype(int)
        audit["order_method"] = ordering["method"]
        audit["order_fallback_reason"] = ordering["fallback_reason"] or ""
        audit["sample_indices"] = json.dumps(sample.tolist())
        audit["sample_positions"] = json.dumps(
            [ordering["positions"][i] for i in sample] if ordering["positions"] is not None else []
        )
        header = ordered_headers[int(sample[5])]
        audit["patient_id_sha256"] = _header_hash(header, ("PatientID",))
        audit["issuer_sha256"] = _header_hash(header, ("IssuerOfPatientID",))
        if str(getattr(header, "PatientID", "")).strip():
            audit["patient_key_sha256"] = _header_hash(header, ("PatientID", "IssuerOfPatientID"))
        audit["scanner_sha256"] = _header_hash(header, (
            "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "StationName", "DeviceSerialNumber",
        ))
        audit["site_sha256"] = _header_hash(header, ("InstitutionName", "InstitutionAddress"))
        removed = str(getattr(header, "PatientIdentityRemoved", "")).upper()
        audit["patient_identity_removed"] = removed if removed in {"YES", "NO"} else "UNKNOWN"
        for key, tag in (("laterality", "Laterality"), ("image_laterality", "ImageLaterality")):
            value = str(getattr(header, tag, "")).upper()
            audit[key] = value if value in {"L", "R", "B", "U"} else "UNKNOWN"
        audit["deidentification_method_present"] = bool(str(getattr(header, "DeidentificationMethod", "")))
        audit["rows"] = int(getattr(header, "Rows", 0))
        audit["columns"] = int(getattr(header, "Columns", 0))
        audit["number_of_frames"] = int(getattr(header, "NumberOfFrames", 1))
        audit["transfer_syntax"] = str(getattr(header.file_meta, "TransferSyntaxUID", ""))
        spacing = _pixel_spacing(header)
        audit["pixel_spacing"] = json.dumps(spacing.tolist() if spacing is not None else [])
        prepared, hashes = [], []
        for index in sample:
            try:
                ds = pydicom.dcmread(ordered_files[int(index)])
                pixels = decode_grayscale(ds)
                pixel_hash = hashlib.sha256(np.asarray(pixels, dtype="<f4").tobytes()).hexdigest()
                spacing = _pixel_spacing(ds)
                if spacing is None:
                    spacing = np.ones(2)
                    audit["spacing_fallbacks"] += 1
                prepared.append(preprocess_slice(pixels, size=224, pixel_spacing=tuple(spacing)))
                hashes.append(pixel_hash)
            except Exception as exc:
                prepared.append(None)
                hashes.append(None)
                audit["decode_failures"] += 1
                errors.add(type(exc).__name__)
        audit["error_types"] = "|".join(sorted(errors))
        good = [i for i, value in enumerate(prepared) if value is not None]
        if not good:
            audit["status"] = "all_sampled_slices_failed"
            audit["sample_pixel_sha256"] = json.dumps(hashes)
            continue
        for i, value in enumerate(prepared):
            if value is None:
                nearest = min(good, key=lambda j: abs(int(sample[j]) - int(sample[i])))
                prepared[i], hashes[i] = prepared[nearest], hashes[nearest]
                audit["repeated_failed_samples"] += 1
        images[plane_index] = np.stack(prepared)
        presence[plane_index] = 1
        audit["sample_pixel_sha256"] = json.dumps(hashes)
        audit["status"] = "usable_with_fallback" if (
            audit["header_failures"] or audit["decode_failures"] or audit["spacing_fallbacks"]
            or ordering["method"] != "geometry"
        ) else "usable"
    quality = {
        "StudyInstanceUID": study, "split": split, "usable_planes": int(presence.sum()),
        "missing_planes": "|".join(plane for plane, present in zip(PLANES, presence) if not present),
        **{key: sum(row[key] for row in audits) for key in (
            "header_failures", "decode_failures", "repeated_failed_samples", "spacing_fallbacks",
        )},
        "instance_number_planes": sum(row["order_method"] == "InstanceNumber" for row in audits),
    }
    return {"images": images, "presence": presence, "quality": quality, "series": audits}


def encode_studies(prepared: list[dict], encode, encoder_batch_size: int = 32):
    """Return float32 plane means, float16 window cache and binary presence.

    Means are computed before float16 cache quantization. All inference uses this
    function too, so caching does not change the mean-pooled experiment recipe.
    """
    if not prepared or encoder_batch_size < 1:
        raise ValueError('Require studies and a positive encoder batch size')
    presence = np.asarray([item['presence'] for item in prepared], dtype=np.float32)
    if presence.shape != (len(prepared), 3) or not np.isin(presence, [0, 1]).all():
        raise ValueError('Invalid plane presence')
    if np.any(presence.sum(axis=1) == 0):
        raise ValueError('Study has zero usable image planes')
    for item in prepared:
        pixels = np.asarray(item['images'])
        if pixels.ndim != 4 or pixels.shape[:2] != (3, 12) or not np.isfinite(pixels).all():
            raise ValueError('Invalid twelve-slice images')
    locations = [(i, plane, window) for i in range(len(prepared))
                 for plane in range(3) if presence[i, plane] for window in range(10)]
    vectors = np.zeros((len(prepared), 3, 10, 768), dtype=np.float32)
    for start in range(0, len(locations), encoder_batch_size):
        batch = locations[start:start + encoder_batch_size]
        pixels = normalize_rgb(np.stack([prepared[i]['images'][plane, window:window + 3]
                                         for i, plane, window in batch]))
        pooled = np.asarray(encode(pixels), dtype=np.float32)
        if (pooled.shape != (len(batch), 768) or not np.isfinite(pooled).all()
                or np.any(np.abs(pooled) > np.finfo(np.float16).max)):
            raise ValueError('Invalid or float16-unrepresentable window embeddings')
        for vector, (i, plane, window) in zip(pooled, batch):
            vectors[i, plane, window] = vector
    means = vectors.mean(axis=2).reshape(len(prepared), 2304)
    features = np.concatenate([means, presence], axis=1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite mean embeddings')
    return features, vectors.astype(np.float16), presence


def _contact_sheet(prepared: list[dict], output: Path) -> None:
    """Show every sampled slice for the first study without patient text."""
    from PIL import Image, ImageDraw

    canvas = Image.new('RGB', (12 * 112, 3 * 132), 'black')
    draw = ImageDraw.Draw(canvas)
    for plane, name in enumerate(PLANES):
        for sample in range(12):
            pixels = prepared[0]['images'][plane, sample]
            tile = Image.fromarray((pixels * 255).clip(0, 255).astype('uint8')).resize((112, 112))
            canvas.paste(tile.convert('RGB'), (sample * 112, plane * 132))
            draw.text((sample * 112 + 2, plane * 132 + 113), f'{name} {sample + 1}', fill='white')
    canvas.save(output / 'preprocessing_contact_sheet.png')


def extract_features(data_root: Path, checkpoint_dir: Path, output_dir: Path, *,
                     batch_size: int = 4, workers: int = 4,
                     encoder_batch_size: int = 32, device: str = 'cuda') -> dict:
    """Stream coverage features and cache, preserving train-then-test input order."""
    import torch
    import transformers
    from transformers import AutoModel

    data_root, checkpoint_dir, output_dir = map(Path, (data_root, checkpoint_dir, output_dir))
    if not 1 <= batch_size <= 12 or not 1 <= workers <= 8 or not 1 <= encoder_batch_size <= 64:
        raise ValueError('Use 1-12 studies, 1-8 workers and 1-64 encoder windows per batch')
    if output_dir.exists():
        raise FileExistsError('Feature output directory already exists')
    provenance = checkpoint_provenance(checkpoint_dir)
    if provenance['model_type'] != 'dinov2' or provenance['hidden_size'] != 384:
        raise ValueError('This fixed experiment requires generic DINOv2 small')
    ids, series_by_split = [], {}
    for split in ('train', 'test'):
        frame = pd.read_csv(data_root / f'{split}.csv', usecols=['StudyInstanceUID'], dtype=str)
        studies = frame['StudyInstanceUID']
        if frame.empty or studies.isna().any() or studies.duplicated().any() or studies.str.strip().eq('').any():
            raise ValueError(f'Invalid {split} study IDs')
        ids.extend((study, split) for study in studies)
        metadata = pd.read_csv(data_root / f'{split}_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
        if not set(metadata['StudyInstanceUID']).issubset(set(studies)):
            raise ValueError('Series metadata contains unknown study IDs')
        series_by_split[split] = {study: group.to_dict('records') for study, group in metadata.groupby('StudyInstanceUID')}
    if len({study for study, _ in ids}) != len(ids):
        raise ValueError('Training and test study IDs overlap')
    model = AutoModel.from_pretrained(str(checkpoint_dir), local_files_only=True,
                                     trust_remote_code=False, weights_only=True).to(device).eval()
    model.requires_grad_(False)
    torch.manual_seed(0)

    def encode(pixels):
        with torch.no_grad():
            tokens = model(pixel_values=torch.from_numpy(pixels).to(device)).last_hidden_state
            return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1).float().cpu().numpy()

    output_dir.mkdir(parents=True)
    pd.DataFrame(ids, columns=['StudyInstanceUID', 'split']).to_csv(output_dir / 'IDs.csv', index=False)
    features = np.lib.format.open_memmap(output_dir / 'features.npy', mode='w+', dtype=np.float32, shape=(len(ids), 2307))
    windows = np.lib.format.open_memmap(output_dir / 'window_features.npy', mode='w+', dtype=np.float16, shape=(len(ids), 3, 10, 768))
    presence = np.lib.format.open_memmap(output_dir / 'presence.npy', mode='w+', dtype=np.uint8, shape=(len(ids), 3))
    pixels = np.lib.format.open_memmap(output_dir / 'pixels_uint8.npy', mode='w+', dtype=np.uint8, shape=(len(ids), 3, 12, 224, 224))
    sources = ('__init__.py', 'coverage.py', 'features.py', 'imaging.py')
    (output_dir / 'source').mkdir()
    for name in sources:
        (output_dir / 'source' / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    manifest = {
        'status': 'running', 'recipe': 'central12-neighbor3-mean10-v1', 'checkpoint': provenance,
        'encoder_frozen': True, 'encoder_fit_on_competition_data': False, 'labels_or_reports_read': False,
        'input_hash_note': 'Entire metadata files hashed as bytes; only study IDs and series metadata parsed.',
        'feature_shape': list(features.shape), 'window_feature_shape': list(windows.shape),
        'window_feature_dtype': 'float16', 'mean_pool_dtype': 'float32 before cache quantization',
        'window_order': [list(range(i, i + 3)) for i in range(10)],
        'embedding_layout': 'CLS[384] then mean-patch[384]',
        'pixel_cache_shape': list(pixels.shape), 'pixel_cache_dtype': 'uint8',
        'pixel_cache_quantization': 'round(255*clip(preprocessed_float,0,1)); cache only; feature extraction uses original float32',
        'presence_shape': list(presence.shape), 'planes': list(PLANES),
        'batch_size': batch_size, 'workers': workers, 'encoder_batch_size': encoder_batch_size,
        'device': device, 'seed': 0, 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'preprocessing': {
            'series_rule': 'plane, fluid-sensitive first, fat-suppressed first, UID tie break',
            'slice_order': 'signed common-normal geometry; complete InstanceNumber fallback with provenance',
            'sampling': '12 evenly spaced slices across central 20%-80%, nearest-index rounding',
            'intensity': 'modality LUT/rescale, MONOCHROME1 inversion, per-slice 1/99 percentile clip',
            'resize': '224 physical-aspect bilinear resize, centered zero padding',
            'channels': 'ten overlapping windows of three consecutive sampled grayscale slices; ImageNet mean/std',
            'laterality': 'header tags audited; no inferred or tag-based flips',
            'pooling': 'CLS concatenated with mean patch tokens per window; arithmetic mean of ten per plane; three presence flags',
            'failed_slice': 'repeat nearest successfully decoded sampled slice; no usable planes rejects study',
        },
        'validation_limit': 'Anonymized patient hashes and sampled pixels do not establish patient independence.',
        'input_sha256': {name: _hash_file(data_root / name) for name in ('train.csv', 'test.csv', 'train_series.csv', 'test_series.csv')},
        'source_sha256': {name: _hash_file(Path(__file__).with_name(name)) for name in sources},
        'completed_studies': 0,
    }
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(ids), batch_size):
                batch = ids[start:start + batch_size]
                futures = [pool.submit(prepare_study, data_root, split, study, series_by_split[split].get(study, []))
                           for study, split in batch]
                prepared = [future.result() for future in futures]
                for name, rows in [('quality.csv', [item['quality'] for item in prepared]),
                                   ('selected_series.csv', [row for item in prepared for row in item['series']])]:
                    pd.DataFrame(rows).to_csv(output_dir / name, index=False, mode='a', header=start == 0)
                values, cache, flags = encode_studies(prepared, encode, encoder_batch_size)
                stop = start + len(batch)
                features[start:stop], windows[start:stop], presence[start:stop] = values, cache, flags
                for i, item in enumerate(prepared, start=start):
                    pixels[i] = np.rint(255 * np.clip(item['images'], 0, 1)).astype(np.uint8)
                for array in (features, windows, presence, pixels):
                    array.flush()
                if start == 0:
                    _contact_sheet(prepared, output_dir)
                    elapsed = time.monotonic() - started
                    manifest['timing'] = {'probe_studies': stop, 'probe_seconds': elapsed,
                                          'estimated_total_seconds': elapsed / stop * len(ids),
                                          'note': 'Initial bounded batch including warmup; throughput estimate only.'}
                    print('Initial throughput estimate: ' + json.dumps(manifest['timing']), flush=True)
                manifest['completed_studies'] = stop
                manifest['elapsed_seconds'] = time.monotonic() - started
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f'Extracted {stop}/{len(ids)} studies; {manifest["elapsed_seconds"]:.1f}s elapsed', flush=True)
        manifest['artifact_sha256'] = {str(path.relative_to(output_dir)): _hash_file(path)
                                       for path in sorted(output_dir.rglob('*'))
                                       if path.is_file() and path != manifest_path}
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest['status'], manifest['error_type'] = 'failed', type(exc).__name__
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract_features(args.data_root, args.checkpoint, args.output)
