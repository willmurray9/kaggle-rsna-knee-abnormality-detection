import json
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from rsnaknee.features import (
    PLANES,
    assemble_features,
    checkpoint_provenance,
    extract_features,
    normalize_rgb,
    prepare_study,
)


@pytest.fixture
def image_study(tmp_path: Path, request):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

    series = []
    for plane_id, plane in enumerate(PLANES, start=1):
        uid = f"1.2.{plane_id}"
        series.append({"StudyInstanceUID": "1.2", "SeriesInstanceUID": uid,
                       "Anatomical_Plane": plane, "Fluid_Sensitive": 1, "Fat_Suppression": 1})
        directory = tmp_path / "train_series" / "1.2" / uid
        directory.mkdir(parents=True)
        n_slices = getattr(request, "param", 3)
        for position in range(n_slices):
            name = ["z", "a", "m"][position] if n_slices == 3 else f"{n_slices - position:02d}"
            ds = pydicom.Dataset()
            ds.file_meta = pydicom.Dataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
            ds.file_meta.MediaStorageSOPInstanceUID = f"{uid}.{position + 1}"
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = f"{uid}.{position + 1}"
            ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2", uid
            ds.PatientID = "synthetic-private-patient"
            ds.IssuerOfPatientID = "synthetic-private-issuer"
            ds.InstitutionName = "synthetic-private-site"
            ds.Manufacturer = "synthetic-scanner"
            ds.PatientIdentityRemoved = "YES"
            ds.Laterality = "L"
            ds.ImageLaterality = "L"
            ds.Rows, ds.Columns = 4, 4
            ds.PixelSpacing = [1, 1]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, position]
            ds.InstanceNumber = 3 - position
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            ds.PixelRepresentation = 0
            pixels = np.roll(np.arange(16, dtype=np.uint16).reshape(4, 4), position, axis=1)
            ds.PixelData = pixels.tobytes()
            pydicom.dcmwrite(directory / f"{name}.dcm", ds, enforce_file_format=True)
    return tmp_path, series


def test_prepare_study_preserves_geometric_channel_order_and_hashes_identifiers(image_study) -> None:
    root, series = image_study
    result = prepare_study(root, "train", "1.2", series)
    assert result["images"].shape == (3, 3, 224, 224)
    assert result["presence"].tolist() == [1, 1, 1]
    assert result["quality"]["missing_planes"] == ""
    selected = result["series"][0]
    assert selected["order_method"] == "geometry"
    assert json.loads(selected["sample_indices"]) == [0, 1, 2]
    assert json.loads(selected["sample_positions"]) == [0.0, 1.0, 2.0]
    assert len(selected["patient_key_sha256"]) == 64
    assert selected["patient_identity_removed"] == "YES"
    assert selected["laterality"] == "L" and selected["image_laterality"] == "L"
    serialized = json.dumps(result["series"])
    assert "synthetic-private" not in serialized
    assert len(set(json.loads(selected["sample_pixel_sha256"]))) == 3


@pytest.mark.parametrize("image_study", [9], indirect=True)
def test_prespecified_central_band_samples_ordered_slices_without_flipping(image_study) -> None:
    root, series = image_study
    selected = prepare_study(root, "train", "1.2", series)["series"][0]
    assert json.loads(selected["sample_indices"]) == [2, 4, 6]
    assert json.loads(selected["sample_positions"]) == [2, 4, 6]


def test_one_corrupt_slice_uses_nearest_sample_with_audited_fallback(image_study) -> None:
    import pydicom

    root, series = image_study
    path = root / "train_series" / "1.2" / "1.2.1" / "a.dcm"
    ds = pydicom.dcmread(path)
    ds.PixelData = b""
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    result = prepare_study(root, "train", "1.2", series)
    assert result["presence"].tolist() == [1, 1, 1]
    assert result["quality"]["decode_failures"] == 1
    assert result["series"][0]["repeated_failed_samples"] == 1
    np.testing.assert_array_equal(result["images"][0, 1], result["images"][0, 0])


def test_corrupt_header_is_excluded_with_provenance(image_study) -> None:
    root, series = image_study
    (root / "train_series" / "1.2" / "1.2.1" / "a.dcm").write_bytes(b"corrupt header")
    result = prepare_study(root, "train", "1.2", series)
    assert result["presence"].tolist() == [1, 1, 1]
    assert result["quality"]["header_failures"] == 1
    assert result["series"][0]["status"] == "usable_with_fallback"


def test_all_failed_samples_mark_plane_missing(image_study) -> None:
    import pydicom

    root, series = image_study
    for path in (root / "train_series" / "1.2" / "1.2.1").glob("*.dcm"):
        ds = pydicom.dcmread(path)
        ds.PixelData = b""
        pydicom.dcmwrite(path, ds, enforce_file_format=True)
    result = prepare_study(root, "train", "1.2", series)
    assert result["presence"].tolist() == [0, 1, 1]
    assert result["series"][0]["status"] == "all_sampled_slices_failed"
    assert result["quality"]["decode_failures"] == 3


def test_missing_plane_is_zero_with_presence_flag_and_reason(image_study) -> None:
    root, series = image_study
    result = prepare_study(root, "train", "1.2", series[:1])
    assert result["presence"].tolist() == [1, 0, 0]
    assert np.all(result["images"][1:] == 0)
    assert result["quality"]["missing_planes"] == "Coronal|Axial"
    assert result["series"][1]["status"] == "no_matching_series"


def test_malformed_spacing_is_audited_and_uses_matrix_aspect(image_study) -> None:
    import pydicom

    root, series = image_study
    path = root / "train_series" / "1.2" / "1.2.1" / "a.dcm"
    ds = pydicom.dcmread(path)
    ds.PixelSpacing = 1
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    result = prepare_study(root, "train", "1.2", series)
    assert result["presence"].tolist() == [1, 1, 1]
    assert result["quality"]["spacing_fallbacks"] == 1


def test_no_usable_plane_is_explicit_for_extraction_to_reject(tmp_path: Path) -> None:
    result = prepare_study(tmp_path, "test", "1.2", [])
    assert result["quality"]["usable_planes"] == 0
    assert not result["presence"].any()


def test_imagenet_normalization_treats_slices_as_rgb_channels() -> None:
    result = normalize_rgb(np.ones((2, 3, 4, 4), dtype=np.float32))
    expected = (1 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(result[0, :, 0, 0], expected, rtol=1e-6)
    assert result.dtype == np.float32


def test_feature_assembly_keeps_study_plane_alignment_and_presence() -> None:
    pooled = np.array([[1, 2, 3, 4], [9, 8, 7, 6], [5, 5, 5, 5]], dtype=np.float32)
    result = assemble_features(pooled, [(1, 2), (0, 1), (1, 0)], 2)
    assert result.shape == (2, 15)
    np.testing.assert_array_equal(result[0], [0, 0, 0, 0, 9, 8, 7, 6, 0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(result[1, -3:], [1, 0, 1])
    with pytest.raises(ValueError):
        assemble_features(pooled, [(0, 1)] * 3, 2)


def test_checkpoint_provenance_hashes_actual_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "dinov2", "hidden_size": 384}')
    (tmp_path / "model.safetensors").write_bytes(b"synthetic model bytes")
    first = checkpoint_provenance(tmp_path)
    assert first["model_type"] == "dinov2"
    assert len(first["files_sha256"]["model.safetensors"]) == 64
    (tmp_path / "model.safetensors").write_bytes(b"changed bytes")
    assert checkpoint_provenance(tmp_path)["files_sha256"] != first["files_sha256"]


@pytest.fixture
def fake_extraction(tmp_path: Path, monkeypatch):
    import pandas as pd
    import rsnaknee.features as module

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __getitem__(self, key):
            return Tensor(self.values[key])

        def to(self, device):
            assert device == "cpu"
            return self

        def mean(self, dim):
            return Tensor(self.values.mean(axis=dim))

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False

        def __call__(self, pixel_values):
            n = len(pixel_values.values)
            tokens = np.empty((n, 5, 384), dtype=np.float32)
            tokens[:, 0] = 2
            tokens[:, 1:] = 4
            return SimpleNamespace(last_hidden_state=Tensor(tokens))

    def load_model(path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="synthetic", manual_seed=lambda _: None, no_grad=nullcontext,
        from_numpy=Tensor, cat=lambda tensors, dim: Tensor(np.concatenate([t.values for t in tensors], axis=dim)),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="synthetic", AutoModel=SimpleNamespace(from_pretrained=load_model),
    ))
    data_root = tmp_path / "data"
    data_root.mkdir()
    for split, ids in [("train", ["b", "a"]), ("test", ["c"])]:
        pd.DataFrame({"StudyInstanceUID": ids}).to_csv(data_root / f"{split}.csv", index=False)
        pd.DataFrame({"StudyInstanceUID": ids, "SeriesInstanceUID": ids}).to_csv(
            data_root / f"{split}_series.csv", index=False,
        )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"dinov2","hidden_size":384}')
    (checkpoint / "model.safetensors").write_bytes(b"test weights")

    def prepare(data_root, split, study, series):
        return {
            "images": np.ones((3, 3, 224, 224), dtype=np.float32),
            "presence": np.array([1, 0, 1]),
            "quality": {"StudyInstanceUID": study, "split": split, "usable_planes": 2},
            "series": [{"StudyInstanceUID": study, "split": split}],
        }

    monkeypatch.setattr(module, "prepare_study", prepare)
    return data_root, checkpoint, tmp_path / "output", prepare


def test_extraction_batches_and_keeps_ids_pooling_and_artifacts_aligned(fake_extraction) -> None:
    import pandas as pd

    data_root, checkpoint, output, _ = fake_extraction
    manifest = extract_features(data_root, checkpoint, output, device="cpu", batch_size=2, workers=1)
    assert manifest["status"] == "complete" and manifest["completed_studies"] == 3
    ids = pd.read_csv(output / "IDs.csv")
    assert ids.to_dict("list") == {"StudyInstanceUID": ["b", "a", "c"], "split": ["train", "train", "test"]}
    features = np.load(output / "features.npy")
    assert features.shape == (3, 2307)
    assert np.all(features[:, :384] == 2) and np.all(features[:, 384:768] == 4)
    assert np.all(features[:, 768:1536] == 0)
    np.testing.assert_array_equal(features[:, -3:], [[1, 0, 1]] * 3)
    assert pd.read_csv(output / "quality.csv")["StudyInstanceUID"].tolist() == ["b", "a", "c"]
    with pytest.raises(FileExistsError):
        extract_features(data_root, checkpoint, output, device="cpu")


def test_zero_usable_study_fails_with_saved_quality_and_manifest(fake_extraction, monkeypatch) -> None:
    import rsnaknee.features as module

    data_root, checkpoint, output, original = fake_extraction

    def prepare(*args):
        result = original(*args)
        result["presence"][:] = 0
        result["quality"]["usable_planes"] = 0
        return result

    monkeypatch.setattr(module, "prepare_study", prepare)
    with pytest.raises(ValueError, match="zero usable"):
        extract_features(data_root, checkpoint, output, device="cpu")
    assert (output / "quality.csv").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
