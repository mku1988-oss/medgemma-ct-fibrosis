"""
CT volume loading with anatomically correct slice order.

Two formats are supported:
  - DICOM series (one .dcm file per slice), as used by OSIC.
  - MetaImage (.mhd + .raw), as used by LUNA16.

Slices are always ordered by their physical position along the body axis,
never by file name or file order. File order varies between scanners, and
relying on it can silently turn "lower lobe" into "upper lobe".

Both loaders return the same three things:
  volume       3D array (slices, rows, columns) in Hounsfield Units (HU)
  z_positions  physical position of each slice in mm, inferior to superior
  metadata     a small dictionary describing the scan
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom

logger = logging.getLogger(__name__)

try:
    import SimpleITK as sitk
except ImportError:  # only needed for LUNA16 .mhd files
    sitk = None

# Slices thicker than this are rejected: fine interstitial detail such as
# reticulation is lost on thick reconstructions.
MAX_SLICE_THICKNESS_MM = 5.0

# Expected HU range for a standard CT reconstruction. Values far outside it
# usually indicate a missing or wrong rescale slope/intercept.
EXPECTED_HU_MIN = -1100
EXPECTED_HU_MAX = 3200


def load_dicom_series(directory: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a DICOM series and return it sorted from inferior to superior.

    Each slice is checked before use: it must be a CT image, it must carry
    ImagePositionPatient (needed for ordering), and it must not be thicker
    than MAX_SLICE_THICKNESS_MM. Pixel values are converted to Hounsfield
    Units with each slice's rescale slope and intercept.

    Args:
        directory: Folder containing the .dcm files of one scan.

    Returns:
        (volume, z_positions, metadata)

    Raises:
        ValueError: If the folder is missing or contains no usable CT slices.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    dicom_files = sorted(directory.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No .dcm files found in {directory}")

    slices = [s for s in (_read_valid_ct_slice(f) for f in dicom_files) if s is not None]
    if not slices:
        raise ValueError(f"No valid CT slices found in {directory}")

    # Sort by physical Z position. Ascending Z runs from feet to head.
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])

    volume = np.stack([_slice_to_hu(s) for s in slices])
    _warn_if_hu_out_of_range(volume, directory.name)

    logger.info("Loaded %s: shape=%s, z=[%.1f, %.1f] mm",
                directory.name, volume.shape, z_positions[0], z_positions[-1])

    return volume, z_positions, _dicom_metadata(slices)


def load_mhd_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a LUNA16 MetaImage (.mhd) volume.

    Physical slice positions are computed from the image origin and spacing.

    Args:
        file_path: Path to the .mhd file (its .raw file must sit beside it).

    Returns:
        (volume, z_positions, metadata)
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for .mhd files: pip install SimpleITK")

    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f"File not found: {file_path}")

    image = sitk.ReadImage(str(file_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (Z, Y, X)

    origin = image.GetOrigin()    # (X, Y, Z) in mm
    spacing = image.GetSpacing()  # (X, Y, Z) in mm
    z_positions = origin[2] + np.arange(volume.shape[0]) * spacing[2]

    metadata = {
        "PatientID": file_path.stem,
        "PixelSpacing": list(spacing[:2]),
        "SliceThickness": spacing[2],
        "Manufacturer": "LUNA16",
    }

    logger.info("Loaded %s: shape=%s, z=[%.1f, %.1f] mm",
                file_path.name, volume.shape, z_positions[0], z_positions[-1])

    return volume, z_positions, metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_valid_ct_slice(path: Path):
    """Read one DICOM file and return it, or None if it fails a check."""
    try:
        dcm = pydicom.dcmread(path)
    except Exception as error:
        logger.warning("Could not read %s: %s", path.name, error)
        return None

    if getattr(dcm, "Modality", "CT") != "CT":
        logger.warning("Skipping non-CT file: %s", path.name)
        return None

    if not hasattr(dcm, "ImagePositionPatient"):
        logger.warning("Skipping file without ImagePositionPatient: %s", path.name)
        return None

    thickness = float(getattr(dcm, "SliceThickness", 0) or 0)
    if thickness > MAX_SLICE_THICKNESS_MM:
        logger.warning("Skipping %.1f mm slice (limit %.1f mm): %s",
                       thickness, MAX_SLICE_THICKNESS_MM, path.name)
        return None

    return dcm


def _slice_to_hu(dcm) -> np.ndarray:
    """Convert stored pixel values to Hounsfield Units."""
    slope = float(getattr(dcm, "RescaleSlope", 1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    return dcm.pixel_array.astype(np.float32) * slope + intercept


def _warn_if_hu_out_of_range(volume: np.ndarray, scan_name: str) -> None:
    """Log a warning when HU values fall well outside the normal CT range."""
    hu_min, hu_max = float(volume.min()), float(volume.max())
    if hu_min < EXPECTED_HU_MIN or hu_max > EXPECTED_HU_MAX:
        logger.warning("HU values outside the expected range in %s: [%.0f, %.0f]",
                       scan_name, hu_min, hu_max)


def _dicom_metadata(slices: list) -> dict:
    """Collect basic scan information from a sorted list of slices."""
    reference = slices[0]
    thicknesses = [float(s.SliceThickness) for s in slices if hasattr(s, "SliceThickness")]
    return {
        "PatientID": getattr(reference, "PatientID", "Unknown"),
        "PixelSpacing": list(getattr(reference, "PixelSpacing", [1.0, 1.0])),
        "SliceThickness": float(np.mean(thicknesses)) if thicknesses else 0.0,
        "Manufacturer": getattr(reference, "Manufacturer", "Unknown"),
    }
