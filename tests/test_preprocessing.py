"""
Tests for CT loading and slice selection.

All tests use small synthetic scans, so they run without the OSIC or
LUNA16 datasets.
"""

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from preprocessing import ct_loader
from preprocessing.ct_processor import CTPreprocessor


# ---------------------------------------------------------------------------
# Synthetic DICOM helpers
# ---------------------------------------------------------------------------

def write_ct_slice(path, z_mm, fill_hu, thickness=1.0, modality="CT"):
    """Write a tiny 8x8 CT slice whose pixels all equal fill_hu."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Modality = modality
    ds.PatientID = "SYNTHETIC"
    ds.ImagePositionPatient = [0.0, 0.0, float(z_mm)]
    ds.SliceThickness = thickness
    ds.Rows, ds.Columns = 8, 8
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    stored = np.full((8, 8), fill_hu + 1024, dtype=np.int16)
    ds.PixelData = stored.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


# ---------------------------------------------------------------------------
# ct_loader
# ---------------------------------------------------------------------------

def test_dicom_slices_are_sorted_by_physical_position(tmp_path):
    # File names deliberately disagree with anatomical order.
    write_ct_slice(tmp_path / "a.dcm", z_mm=30, fill_hu=-300)
    write_ct_slice(tmp_path / "b.dcm", z_mm=10, fill_hu=-100)
    write_ct_slice(tmp_path / "c.dcm", z_mm=20, fill_hu=-200)

    volume, z_positions, _ = ct_loader.load_dicom_series(tmp_path)

    assert list(z_positions) == [10, 20, 30]
    assert [volume[i, 0, 0] for i in range(3)] == [-100, -200, -300]


def test_rescale_gives_hounsfield_units(tmp_path):
    write_ct_slice(tmp_path / "s.dcm", z_mm=0, fill_hu=-850)
    volume, _, _ = ct_loader.load_dicom_series(tmp_path)
    assert np.all(volume == -850)


def test_thick_and_non_ct_slices_are_skipped(tmp_path):
    write_ct_slice(tmp_path / "ok.dcm", z_mm=0, fill_hu=-800)
    write_ct_slice(tmp_path / "thick.dcm", z_mm=5, fill_hu=-800, thickness=7.0)
    write_ct_slice(tmp_path / "mr.dcm", z_mm=10, fill_hu=-800, modality="MR")

    volume, z_positions, _ = ct_loader.load_dicom_series(tmp_path)

    assert volume.shape[0] == 1
    assert list(z_positions) == [0]


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        ct_loader.load_dicom_series(tmp_path)


# ---------------------------------------------------------------------------
# CTPreprocessor
# ---------------------------------------------------------------------------

def test_lower_lobe_region_uses_physical_range():
    processor = CTPreprocessor(slice_region=(0.15, 0.40))
    z_positions = np.arange(0, 101, 10.0)  # 0, 10, ..., 100 mm

    assert processor.select_lower_lobe_indices(z_positions) == [2, 3, 4]


def test_lung_window_scales_to_unit_range():
    processor = CTPreprocessor(lung_window=(-1400, 240))
    result = processor.apply_lung_window(np.array([-2000.0, -1400.0, -580.0, 240.0, 1000.0]))

    np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_lung_fraction_counts_lung_range_pixels():
    ct_slice = np.full((10, 10), 40.0)  # soft tissue
    ct_slice[:5] = -800.0               # half the slice is lung
    assert CTPreprocessor.lung_fraction(ct_slice) == pytest.approx(0.5)


def test_slices_near_nodules_are_excluded():
    processor = CTPreprocessor(nodule_margin_mm=20.0)
    z_positions = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
    annotations = pd.DataFrame({
        "seriesuid": ["scan1"],
        "coordZ": [40.0],
        "diameter_mm": [10.0],  # exclusion distance = 5 + 20 = 25 mm
    })

    kept = processor.exclude_nodule_slices("scan1", [0, 1, 2, 3, 4], z_positions, annotations)

    assert kept == [0, 4]


def test_scan_without_nodules_keeps_all_slices():
    processor = CTPreprocessor()
    annotations = pd.DataFrame({"seriesuid": ["other"], "coordZ": [0.0], "diameter_mm": [5.0]})

    kept = processor.exclude_nodule_slices("scan1", [0, 1], np.array([0.0, 10.0]), annotations)

    assert kept == [0, 1]


def test_process_volume_end_to_end(tmp_path):
    # 20 lung-like slices from 0 to 95 mm; the lower-lobe band is 14.25 to 38 mm.
    for i in range(20):
        write_ct_slice(tmp_path / f"{i:02d}.dcm", z_mm=i * 5, fill_hu=-800)

    result = CTPreprocessor().process_volume(str(tmp_path), scan_id="synthetic")

    kept_z = [s["z_mm"] for s in result["slices"]]
    assert kept_z == [15.0, 20.0, 25.0, 30.0, 35.0]
    assert all(0.0 <= s["image"].min() and s["image"].max() <= 1.0 for s in result["slices"])
