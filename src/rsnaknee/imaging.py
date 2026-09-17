"""Small image helpers for new experiments, with explicit failure and ordering provenance.

These are not a replacement for a checkpoint's original preprocessing. Imported
weights must retain their own normalization, crop, resize and sampling recipe.
Install the imaging extra for DICOM decoding and resizing. No files are modified.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def order_slices(headers: Sequence[Any]) -> tuple[list[int], dict[str, Any]]:
    """Order single-frame DICOM headers along a shared signed slice normal.

    Read every slice header with pydicom.dcmread(..., stop_before_pixels=True).
    All slices use geometry, or all use unique InstanceNumber values; never mix
    millimeters and instance numbers. Returned positions follow the returned order.
    Inconsistent valid orientations indicate a mixed series and are rejected.
    InstanceNumber fallback does not establish physical ordering or laterality.
    """
    if not headers:
        raise ValueError("Cannot order an empty series")
    orientations, positions = [], []
    reason = None
    for ds in headers:
        try:
            orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
            position = np.asarray(ds.ImagePositionPatient, dtype=float)
            valid = (
                orientation.shape == (6,)
                and position.shape == (3,)
                and np.isfinite(orientation).all()
                and np.isfinite(position).all()
            )
            if not valid:
                raise ValueError("Invalid geometry")
            row, col = orientation[:3], orientation[3:]
            if not (
                np.isclose(np.linalg.norm(row), 1, atol=1e-3)
                and np.isclose(np.linalg.norm(col), 1, atol=1e-3)
                and np.isclose(np.dot(row, col), 0, atol=1e-3)
            ):
                raise ValueError("Invalid direction cosines")
            orientations.append(orientation)
            positions.append(position)
        except (AttributeError, TypeError, ValueError):
            reason = "missing or invalid geometry"
    if orientations and not np.allclose(orientations, orientations[0], atol=1e-3):
        raise ValueError("Inconsistent slice orientation; inspect or separate this series")
    if reason is None:
        normal = np.cross(orientations[0][:3], orientations[0][3:])
        normal /= np.linalg.norm(normal)
        keys = np.asarray(positions) @ normal
        if len(keys) > 1 and np.any(np.diff(np.sort(keys)) <= 1e-4):
            reason = "duplicate slice positions; possible multi-echo or repeated slices"
        else:
            indices = np.argsort(keys).tolist()
            return indices, {
                "method": "geometry",
                "positions": keys[indices].tolist(),
                "normal": normal.tolist(),
                "fallback_reason": None,
            }
    try:
        instances = np.array([float(ds.InstanceNumber) for ds in headers])
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot order series: {reason}; InstanceNumber unavailable") from exc
    if not np.isfinite(instances).all() or len(np.unique(instances)) != len(instances):
        raise ValueError(f"Cannot order series: {reason}; InstanceNumber not finite and unique")
    indices = np.argsort(instances).tolist()
    return indices, {
        "method": "InstanceNumber",
        "positions": None,
        "normal": None,
        "fallback_reason": reason,
    }


def decode_grayscale(dataset: Any) -> np.ndarray:
    """Decode one grayscale frame, apply modality LUT/rescale and MONOCHROME1.

    Decoder failures propagate to the caller for explicit missing-series handling.
    Additional pydicom decoder plugins may be required by compressed transfer syntaxes.
    """
    from pydicom.pixels import apply_modality_lut

    photo = str(getattr(dataset, "PhotometricInterpretation", ""))
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Expected MONOCHROME1 or MONOCHROME2")
    if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
        raise ValueError("Expected one grayscale sample per pixel")
    if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
        raise ValueError("Multi-frame DICOM requires separate per-frame geometry handling")
    pixels = np.asarray(apply_modality_lut(dataset.pixel_array, dataset), dtype=np.float32)
    if pixels.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all():
        raise ValueError("Expected a finite, nonempty 2D grayscale image")
    if photo == "MONOCHROME1":
        pixels = pixels.min() + pixels.max() - pixels
    return pixels


def preprocess_slice(
    pixels: np.ndarray,
    size: int = 224,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Clip per-slice 1st/99th percentiles, resize and center-pad into [0, 1].

    PixelSpacing is (row_mm, column_mm); pass it to preserve physical aspect ratio.
    The default preserves matrix aspect ratio when spacing is unavailable. This
    performs no learned fit, anatomical reorientation, cropping or encoder-specific
    channel normalization. Record spacing availability in the experiment audit.
    """
    from PIL import Image

    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all():
        raise ValueError("Expected a finite, nonempty 2D grayscale image")
    spacing = np.asarray(pixel_spacing, dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Pixel spacing must contain two positive finite numbers")
    if not isinstance(size, int) or size < 1:
        raise ValueError("Output size must be a positive integer")
    low, high = np.percentile(pixels, [1, 99])
    normalized = np.zeros_like(pixels) if high <= low else np.clip((pixels - low) / (high - low), 0, 1)
    physical_shape = np.array(pixels.shape) * spacing
    height, width = np.maximum(1, np.rint(physical_shape * size / physical_shape.max()).astype(int))
    resized = np.asarray(
        Image.fromarray(normalized.astype(np.float32)).resize((int(width), int(height)), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    result = np.zeros((size, size), dtype=np.float32)
    top, left = (size - height) // 2, (size - width) // 2
    result[top:top + height, left:left + width] = resized
    return result


def sample_slice_indices(n_slices: int, count: int) -> np.ndarray:
    """Evenly sample the entire ordered stack; repeat indices for short stacks."""
    if n_slices < 1 or count < 1:
        raise ValueError("Slice and sample counts must be positive")
    if count == 1:
        return np.array([n_slices // 2], dtype=int)
    return np.rint(np.linspace(0, n_slices - 1, count)).astype(int)


def choose_series(
    rows: Sequence[Mapping[str, Any]], plane: str = "Sagittal"
) -> Mapping[str, Any] | None:
    """Select one plane, prefer fluid-sensitive then fat-suppressed, tie by UID.

    Pass metadata records for exactly one study. An absent plane returns None for
    explicit missing-series handling. Missing flags never count as positive.
    """
    studies = {row.get("StudyInstanceUID") for row in rows}
    if len(studies) > 1:
        raise ValueError("Series selection requires records for exactly one study")
    candidates = [row for row in rows if str(row.get("Anatomical_Plane", "")).casefold() == plane.casefold()]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            str(row.get("Fluid_Sensitive")) not in {"1", "1.0", "True"},
            str(row.get("Fat_Suppression")) not in {"1", "1.0", "True"},
            str(row["SeriesInstanceUID"]),
        ),
    )
