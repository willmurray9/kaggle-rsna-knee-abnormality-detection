from types import SimpleNamespace

import numpy as np
import pytest

from rsnaknee.imaging import (
    choose_series,
    decode_grayscale,
    order_slices,
    preprocess_slice,
    sample_slice_indices,
)


def header(x: float, instance: int) -> SimpleNamespace:
    return SimpleNamespace(
        ImageOrientationPatient=[0, 1, 0, 0, 0, -1],
        ImagePositionPatient=[x, 0, 0],
        InstanceNumber=instance,
    )


def test_order_projects_onto_signed_normal_instead_of_filename_or_instance() -> None:
    headers = [header(1, 1), header(3, 3), header(2, 2)]
    indices, provenance = order_slices(headers)
    assert indices == [1, 2, 0]  # The normal points toward negative x.
    assert provenance["method"] == "geometry"
    assert provenance["positions"] == [-3.0, -2.0, -1.0]


def test_partial_geometry_uses_instance_for_entire_series_with_reason() -> None:
    headers = [header(100, 2), header(200, 1)]
    del headers[1].ImagePositionPatient
    indices, provenance = order_slices(headers)
    assert indices == [1, 0]
    assert provenance["method"] == "InstanceNumber"
    assert provenance["fallback_reason"]


def test_oblique_order_uses_projection_not_one_patient_axis() -> None:
    root_half = np.sqrt(0.5)
    headers = [header(0, 1), header(0, 2)]
    for ds in headers:
        ds.ImageOrientationPatient = [root_half, root_half, 0, 0, 0, 1]
    headers[0].ImagePositionPatient = [10, 11, 0]
    headers[1].ImagePositionPatient = [0, 0, 0]
    indices, provenance = order_slices(headers)
    assert indices == [0, 1]
    assert provenance["positions"][0] == pytest.approx(-root_half)


def test_inconsistent_orientation_is_rejected() -> None:
    headers = [header(1, 1), header(2, 2)]
    headers[1].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    with pytest.raises(ValueError, match="orientation"):
        order_slices(headers)


def test_duplicate_positions_are_explicit_in_fallback_provenance() -> None:
    _, provenance = order_slices([header(1, 2), header(1, 1)])
    assert provenance["method"] == "InstanceNumber"
    assert "duplicate" in provenance["fallback_reason"]


@pytest.mark.parametrize("headers", [[], [SimpleNamespace()], [SimpleNamespace(InstanceNumber=1)] * 2])
def test_unorderable_series_never_uses_file_order(headers: list) -> None:
    with pytest.raises(ValueError):
        order_slices(headers)


def dicom(pixels: np.ndarray, photo: str = "MONOCHROME2"):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.uid import ExplicitVRLittleEndian

    ds = pydicom.Dataset()
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photo
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = pixels.astype("<i2").tobytes()
    return ds


def test_signed_dicom_pixels_apply_modality_rescale() -> None:
    ds = dicom(np.array([[-2, 0], [1, 3]]))
    ds.RescaleSlope = 2
    ds.RescaleIntercept = -10
    result = decode_grayscale(ds)
    np.testing.assert_array_equal(result, [[-14, -10], [-8, -4]])
    assert result.dtype == np.float32


def test_monochrome1_is_inverted_after_rescale() -> None:
    ds = dicom(np.array([[0, 1], [2, 3]]), "MONOCHROME1")
    ds.RescaleSlope = 2
    ds.RescaleIntercept = -10
    np.testing.assert_array_equal(decode_grayscale(ds), [[-4, -6], [-8, -10]])


def test_zero_rescale_slope_is_not_replaced_with_default() -> None:
    ds = dicom(np.array([[0, 1], [2, 3]]))
    ds.RescaleSlope = 0
    ds.RescaleIntercept = 5
    np.testing.assert_array_equal(decode_grayscale(ds), np.full((2, 2), 5))


def test_invalid_grayscale_and_decode_errors_are_not_black_images() -> None:
    ds = dicom(np.ones((2, 2)))
    ds.PhotometricInterpretation = "RGB"
    with pytest.raises(ValueError, match="MONOCHROME"):
        decode_grayscale(ds)
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = b""
    with pytest.raises(ValueError):
        decode_grayscale(ds)


def test_resize_preserves_physical_aspect_ratio_and_normalizes() -> None:
    pixels = np.tile(np.arange(4), (2, 1)).astype(float)
    result = preprocess_slice(pixels, size=8)
    assert result.shape == (8, 8)
    assert result.dtype == np.float32
    assert np.all(result[:2] == 0) and np.all(result[6:] == 0)
    assert result[2:6].max() == 1
    # Twice the row spacing makes this matrix physically square.
    square = preprocess_slice(pixels, size=8, pixel_spacing=(2, 1))
    assert square[0].max() == 1 and square[-1].max() == 1


def test_normalization_clips_outliers_and_handles_constant_input() -> None:
    pixels = np.arange(10000, dtype=float).reshape(100, 100)
    pixels[-1, -1] = 1e9
    result = preprocess_slice(pixels, size=100)
    assert result[50, 0] == pytest.approx(0.5, abs=0.01)
    assert result.max() == 1 and result.min() == 0
    assert np.all(preprocess_slice(np.ones((2, 2)), size=4) == 0)


@pytest.mark.parametrize("pixels", [np.array([[np.nan]]), np.zeros((1, 2, 3)), np.empty((0, 2))])
def test_invalid_pixels_fail_explicitly(pixels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        preprocess_slice(pixels)


def test_slice_sample_is_fixed_size_centered_and_deterministic() -> None:
    assert sample_slice_indices(9, 3).tolist() == [0, 4, 8]
    assert sample_slice_indices(9, 1).tolist() == [4]
    assert sample_slice_indices(2, 5).tolist() == [0, 0, 0, 1, 1]
    with pytest.raises(ValueError):
        sample_slice_indices(0, 3)


def test_series_rule_is_stable_and_does_not_substitute_other_planes() -> None:
    rows = [
        {"SeriesInstanceUID": "b", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        {"SeriesInstanceUID": "a", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        {"SeriesInstanceUID": "c", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 0, "Fat_Suppression": 1},
    ]
    assert choose_series(rows)["SeriesInstanceUID"] == "a"
    assert choose_series(rows[::-1])["SeriesInstanceUID"] == "a"
    assert choose_series(rows, plane="Coronal") is None
