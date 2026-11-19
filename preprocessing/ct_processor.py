"""
Slice selection and lung windowing for pulmonary fibrosis detection.

Pipeline for one CT volume:
  1. Load the volume in physical slice order (see ct_loader.py).
  2. Keep slices from the lower lung, where idiopathic pulmonary fibrosis
     typically begins (basal predominance).
  3. Drop slices that contain too little lung (diaphragm, abdomen, air).
  4. For the normal class (LUNA16), drop slices close to annotated nodules,
     so that "normal" images really show normal lung.
  5. Apply the lung window and scale to the range 0 to 1.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import ct_loader

logger = logging.getLogger(__name__)


class CTPreprocessor:
    """
    Turns a CT volume into a list of windowed lower-lobe slices.

    Args:
        lung_window: (min HU, max HU). The default -1400 to 240 HU shows lung
            parenchyma, airways and fibrotic patterns such as reticulation
            and honeycombing.
        slice_region: Fraction of the scanned Z range to keep, measured from
            the inferior end (0.0 = lowest slice). The default 0.15 to 0.40
            covers the lower lobes while skipping the diaphragm region.
        min_lung_fraction: Minimum fraction of pixels in the lung HU range
            (-1000 to -200) for a slice to be kept.
        nodule_margin_mm: Extra distance added to a nodule's radius when
            excluding nearby slices.
    """

    def __init__(self,
                 lung_window: Tuple[int, int] = (-1400, 240),
                 slice_region: Tuple[float, float] = (0.15, 0.40),
                 min_lung_fraction: float = 0.25,
                 nodule_margin_mm: float = 20.0):
        self.lung_window = lung_window
        self.slice_region = slice_region
        self.min_lung_fraction = min_lung_fraction
        self.nodule_margin_mm = nodule_margin_mm

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def process_volume(self,
                       input_path: str,
                       scan_id: str,
                       nodule_annotations: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Run the full pipeline on one scan.

        Args:
            input_path: DICOM folder (OSIC) or .mhd file (LUNA16).
            scan_id: Identifier used in logs and output file names.
            nodule_annotations: LUNA16 annotations table (columns seriesuid,
                coordZ, diameter_mm). Pass it only for normal-class scans.

        Returns:
            {"scan_id", "metadata", "slices"} where each slice is a dict with
            "image" (2D float array, 0 to 1), "slice_index" and "z_mm".
            Returns None if no slice passes the checks.
        """
        volume, z_positions, metadata = self.load_volume(input_path)

        indices = self.select_lower_lobe_indices(z_positions)
        indices = [i for i in indices if self.lung_fraction(volume[i]) >= self.min_lung_fraction]

        if nodule_annotations is not None:
            indices = self.exclude_nodule_slices(scan_id, indices, z_positions, nodule_annotations)

        if not indices:
            logger.warning("No usable slices for %s", scan_id)
            return None

        slices = [{
            "image": self.apply_lung_window(volume[i]),
            "slice_index": int(i),
            "z_mm": float(z_positions[i]),
        } for i in indices]

        return {"scan_id": scan_id, "metadata": metadata, "slices": slices}

    # -----------------------------------------------------------------------
    # Individual steps
    # -----------------------------------------------------------------------

    @staticmethod
    def load_volume(input_path: str):
        """Load a DICOM folder or a .mhd file, whichever the path points to."""
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == ".mhd":
            return ct_loader.load_mhd_file(str(path))
        if path.is_dir():
            return ct_loader.load_dicom_series(str(path))
        raise ValueError(f"Expected a DICOM folder or a .mhd file: {input_path}")

    def select_lower_lobe_indices(self, z_positions: np.ndarray) -> List[int]:
        """
        Return indices of slices inside the lower-lung region.

        The region is defined on physical Z positions, which must already be
        sorted from inferior to superior.
        """
        if len(z_positions) == 0:
            return []

        z_min, z_max = float(z_positions[0]), float(z_positions[-1])
        z_range = z_max - z_min
        lower = z_min + z_range * self.slice_region[0]
        upper = z_min + z_range * self.slice_region[1]

        return [i for i, z in enumerate(z_positions) if lower <= z <= upper]

    @staticmethod
    def lung_fraction(ct_slice: np.ndarray) -> float:
        """Fraction of pixels in the lung HU range (-1000 to -200)."""
        return float(np.mean((ct_slice > -1000) & (ct_slice < -200)))

    def exclude_nodule_slices(self,
                              scan_id: str,
                              indices: List[int],
                              z_positions: np.ndarray,
                              annotations: pd.DataFrame) -> List[int]:
        """
        Remove slices within (nodule radius + margin) of any annotated nodule.

        LUNA16 scans come from a lung cancer screening cohort. Removing the
        slices near known nodules keeps the normal class free of focal
        pathology.
        """
        nodules = annotations[annotations["seriesuid"] == scan_id]
        if nodules.empty:
            return indices

        safe = []
        for i in indices:
            distances = np.abs(nodules["coordZ"].to_numpy() - z_positions[i])
            limits = nodules["diameter_mm"].to_numpy() / 2 + self.nodule_margin_mm
            if np.all(distances >= limits):
                safe.append(i)
        return safe

    def apply_lung_window(self, ct_slice: np.ndarray) -> np.ndarray:
        """Clip to the lung window and scale to the range 0 to 1."""
        window_min, window_max = self.lung_window
        clipped = np.clip(ct_slice, window_min, window_max)
        return (clipped - window_min) / (window_max - window_min)
