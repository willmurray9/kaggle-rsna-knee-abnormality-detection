"""Stream frozen generic DINOv2 features and a privacy-preserving image audit.

The three RGB channels are three physically ordered grayscale slices. This is an
explicit new experiment recipe, not preprocessing for a competition-trained model.
Torch and transformers are needed only on the feature extraction host.
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

from rsnaknee.imaging import choose_series, decode_grayscale, order_slices, preprocess_slice


PLANES = ("Sagittal", "Coronal", "Axial")
HEADER_TAGS = [
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "InstanceNumber",
    "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns",
    "NumberOfFrames", "PatientID", "IssuerOfPatientID", "PatientIdentityRemoved",
    "DeidentificationMethod", "InstitutionName", "InstitutionAddress", "StationName",
    "DeviceSerialNumber", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "Laterality", "ImageLaterality",
]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_hash(header: Any, tags: tuple[str, ...]) -> str:
    values = [str(getattr(header, tag, "")).strip() for tag in tags]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest() if any(values) else ""


def _pixel_spacing(header: Any) -> np.ndarray | None:
    try:
        spacing = np.asarray(getattr(header, "PixelSpacing", []), dtype=float)
    except (TypeError, ValueError):
        return None
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        return None
    return spacing


def checkpoint_provenance(checkpoint: Path) -> dict[str, Any]:
    """Hash local config and weights without contacting a model registry."""
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    files = [path for path in sorted(checkpoint.rglob("*"))
             if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".pt", ".pth"}]
    if not any(path.suffix in {".safetensors", ".bin", ".pt", ".pth"} for path in files):
        raise ValueError("Checkpoint directory has no local model weights")
    return {
        "directory": str(checkpoint),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "files_sha256": {str(path.relative_to(checkpoint)): _hash_file(path) for path in files},
    }


def prepare_study(data_root: Path, split: str, study: str, series: list[dict]) -> dict[str, Any]:
    """Read one selected series per plane; report partial failures instead of hiding them."""
    import pydicom

    images = np.zeros((len(PLANES), 3, 224, 224), dtype=np.float32)
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
        sample = np.rint(np.linspace(0.2 * (len(indices) - 1), 0.8 * (len(indices) - 1), 3)).astype(int)
        audit["order_method"] = ordering["method"]
        audit["order_fallback_reason"] = ordering["fallback_reason"] or ""
        audit["sample_indices"] = json.dumps(sample.tolist())
        audit["sample_positions"] = json.dumps(
            [ordering["positions"][i] for i in sample] if ordering["positions"] is not None else []
        )
        header = ordered_headers[int(sample[1])]
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


def normalize_rgb(images: np.ndarray) -> np.ndarray:
    """Apply DINOv2's ImageNet channel normalization to 2.5D images in [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return (np.asarray(images, dtype=np.float32) - mean) / std


def assemble_features(pooled: np.ndarray, locations: list[tuple[int, int]], n_studies: int) -> np.ndarray:
    """Place CLS+mean-patch vectors into fixed plane slots, then append presence flags."""
    if pooled.ndim != 2 or len(pooled) != len(locations) or not np.isfinite(pooled).all():
        raise ValueError("Invalid pooled embeddings")
    if len(set(locations)) != len(locations):
        raise ValueError("Duplicate study/plane embedding")
    width = pooled.shape[1]
    features = np.zeros((n_studies, len(PLANES) * width + len(PLANES)), dtype=np.float32)
    for vector, (study, plane) in zip(pooled, locations):
        if not 0 <= study < n_studies or not 0 <= plane < len(PLANES):
            raise ValueError("Embedding location outside batch")
        features[study, plane * width:(plane + 1) * width] = vector
        features[study, len(PLANES) * width + plane] = 1
    return features


def extract_features(
    data_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = 12,
    workers: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    """Extract all train/test rows without reading reports, labels or using the network."""
    import torch
    import transformers
    from transformers import AutoModel

    data_root, checkpoint_dir, output_dir = map(Path, (data_root, checkpoint_dir, output_dir))
    if batch_size < 1 or not 1 <= workers <= 8:
        raise ValueError("Use a positive batch size and 1 to 8 workers")
    if output_dir.exists():
        raise FileExistsError("Feature output directory already exists")
    provenance = checkpoint_provenance(checkpoint_dir)
    if provenance["model_type"] != "dinov2" or provenance["hidden_size"] != 384:
        raise ValueError("This fixed experiment requires generic DINOv2 small (hidden_size=384)")
    ids, series_by_split = [], {}
    for split in ("train", "test"):
        frame = pd.read_csv(data_root / f"{split}.csv", usecols=["StudyInstanceUID"], dtype=str)
        if frame.empty or frame["StudyInstanceUID"].isna().any() or frame["StudyInstanceUID"].duplicated().any():
            raise ValueError(f"Invalid {split} study IDs")
        ids.extend((study, split) for study in frame["StudyInstanceUID"])
        metadata = pd.read_csv(data_root / f"{split}_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
        series_by_split[split] = {study: group.to_dict("records") for study, group in metadata.groupby("StudyInstanceUID")}
    if len({study for study, _ in ids}) != len(ids):
        raise ValueError("Training and test study IDs overlap")
    model = AutoModel.from_pretrained(
        str(checkpoint_dir), local_files_only=True, trust_remote_code=False,
    ).to(device).eval()
    model.requires_grad_(False)
    torch.manual_seed(0)
    output_dir.mkdir(parents=True)
    pd.DataFrame(ids, columns=["StudyInstanceUID", "split"]).to_csv(output_dir / "IDs.csv", index=False)
    features = np.lib.format.open_memmap(output_dir / "features.npy", mode="w+", dtype=np.float32, shape=(len(ids), 2307))
    started = time.monotonic()
    manifest = {
        "status": "running", "checkpoint": provenance, "encoder_frozen": True,
        "encoder_fit_on_competition_data": False, "labels_or_reports_read": False,
        "feature_shape": list(features.shape), "planes": list(PLANES), "batch_size": batch_size,
        "workers": workers, "device": device, "seed": 0,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "preprocessing": {
            "series_rule": "plane, fluid-sensitive first, fat-suppressed first, UID tie break",
            "slice_order": "signed common-normal geometry; complete InstanceNumber fallback with provenance",
            "sampling": "3 evenly spaced slices across central 20%-80% of ordered stack, rounded to nearest index",
            "intensity": "modality LUT/rescale, MONOCHROME1 inversion, per-slice 1/99 percentile clip",
            "resize": "224 physical-aspect bilinear resize, centered zero padding",
            "channels": "three ordered grayscale slices as RGB; ImageNet mean/std",
            "laterality": "header tags audited; no inferred or tag-based flips",
            "pooling": "CLS concatenated with mean patch tokens per plane; then three presence flags",
            "failed_slice": "repeat nearest successfully decoded sampled slice; no usable planes rejects study",
        },
        "validation_limit": "Hashed anonymized patient/site identifiers do not establish patient independence.",
        "input_sha256": {name: _hash_file(data_root / name) for name in (
            "train.csv", "test.csv", "train_series.csv", "test_series.csv",
        )},
        "source_sha256": {name: _hash_file(Path(__file__).with_name(name)) for name in ("features.py", "imaging.py")},
        "completed_studies": 0,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start + batch_size]
                futures = [pool.submit(prepare_study, data_root, split, study, series_by_split[split].get(study, []))
                           for study, split in batch_ids]
                prepared = [future.result() for future in futures]
                pd.DataFrame([item["quality"] for item in prepared]).to_csv(
                    output_dir / "quality.csv", index=False, mode="a", header=start == 0,
                )
                pd.DataFrame([row for item in prepared for row in item["series"]]).to_csv(
                    output_dir / "selected_series.csv", index=False, mode="a", header=start == 0,
                )
                if any(not item["presence"].any() for item in prepared):
                    raise ValueError("Study has zero usable image planes; inspect quality.csv")
                locations = [(i, plane) for i, item in enumerate(prepared)
                             for plane, present in enumerate(item["presence"]) if present]
                pixels = normalize_rgb(np.stack([prepared[i]["images"][plane] for i, plane in locations]))
                with torch.no_grad():
                    tokens = model(pixel_values=torch.from_numpy(pixels).to(device)).last_hidden_state
                    pooled = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1).float().cpu().numpy()
                features[start:start + len(prepared)] = assemble_features(pooled, locations, len(prepared))
                features.flush()
                manifest["completed_studies"] = start + len(prepared)
                manifest["elapsed_seconds"] = time.monotonic() - started
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"Extracted {manifest['completed_studies']}/{len(ids)} studies", flush=True)
        manifest["status"] = "complete"
        manifest["artifact_sha256"] = {name: _hash_file(output_dir / name) for name in (
            "IDs.csv", "features.npy", "quality.csv", "selected_series.csv",
        )}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_features(args.data_root, args.checkpoint, args.output,
                     batch_size=args.batch_size, workers=args.workers, device=args.device)


if __name__ == "__main__":
    main()
