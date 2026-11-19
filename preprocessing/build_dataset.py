"""
Build the slice dataset from raw OSIC and LUNA16 scans.

For every scan, lower-lobe slices are selected and windowed (see
ct_processor.py), up to --slices-per-patient of them are sampled at random,
and each is saved as an RGB PNG. A manifest lists every saved slice:

    [{"patient_id": "...", "label": 1, "image": "images/<patient>_slice_<n>.png"}, ...]

Labels: 1 = fibrosis (OSIC), 0 = normal (LUNA16).

Usage:
    python -m preprocessing.build_dataset \
        --osic-dir  path/to/osic/train \
        --luna-dir  path/to/luna16 \
        --luna-annotations path/to/luna16/annotations.csv \
        --output-dir data/processed
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .ct_processor import CTPreprocessor

logger = logging.getLogger(__name__)

FIBROSIS, NORMAL = 1, 0


def build_dataset(osic_dir: Optional[Path],
                  luna_dir: Optional[Path],
                  luna_annotations: Optional[Path],
                  output_dir: Path,
                  slices_per_patient: int = 10,
                  seed: int = 42) -> List[Dict]:
    """
    Process both datasets and write the PNG slices and manifest.json.

    Returns:
        The manifest entries that were written.
    """
    random.seed(seed)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    processor = CTPreprocessor()
    annotations = pd.read_csv(luna_annotations) if luna_annotations else None
    if luna_dir and annotations is None:
        logger.warning("No LUNA16 annotations given: nodule slices will NOT be excluded.")

    manifest: List[Dict] = []
    for scan_path, scan_id, label in _iter_scans(osic_dir, luna_dir):
        nodules = annotations if label == NORMAL else None
        try:
            result = processor.process_volume(str(scan_path), scan_id, nodules)
        except Exception as error:
            logger.error("Failed to process %s: %s", scan_id, error)
            continue
        if result is None:
            continue

        chosen = _sample(result["slices"], slices_per_patient)
        for s in chosen:
            relative_path = _save_png(s["image"], image_dir, scan_id, s["slice_index"])
            manifest.append({"patient_id": scan_id, "label": label, "image": relative_path})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _log_summary(manifest, manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_scans(osic_dir: Optional[Path],
                luna_dir: Optional[Path]) -> Iterable[Tuple[Path, str, int]]:
    """Yield (path, scan_id, label) for every scan, with a progress bar."""
    scans: List[Tuple[Path, str, int]] = []
    if osic_dir:
        # One folder of DICOM files per OSIC patient.
        scans += [(p, p.name, FIBROSIS) for p in sorted(osic_dir.iterdir()) if p.is_dir()]
    if luna_dir:
        # One .mhd file per LUNA16 scan, spread over subset folders.
        scans += [(p, p.stem, NORMAL) for p in sorted(luna_dir.rglob("*.mhd"))]
    return tqdm(scans, desc="Scans")


def _sample(slices: List[Dict], k: int) -> List[Dict]:
    """Pick up to k slices at random, keeping their anatomical order."""
    if len(slices) <= k:
        return slices
    return sorted(random.sample(slices, k), key=lambda s: s["slice_index"])


def _save_png(image: np.ndarray, image_dir: Path, scan_id: str, slice_index: int) -> str:
    """
    Save a 0 to 1 float slice as an 8-bit RGB PNG.

    MedGemma's image processor expects three channels, so the grayscale slice
    is repeated in R, G and B.
    """
    pixels = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    filename = f"{scan_id}_slice_{slice_index}.png"
    Image.fromarray(pixels).convert("RGB").save(image_dir / filename)
    return f"{image_dir.name}/{filename}"


def _log_summary(manifest: List[Dict], path: Path) -> None:
    patients = {(m["patient_id"], m["label"]) for m in manifest}
    n_fib = sum(1 for _, label in patients if label == FIBROSIS)
    logger.info("Wrote %d slices from %d patients (%d fibrosis, %d normal) to %s",
                len(manifest), len(patients), n_fib, len(patients) - n_fib, path)


def main():
    parser = argparse.ArgumentParser(description="Build the CT slice dataset.")
    parser.add_argument("--osic-dir", type=Path, help="OSIC train folder (one subfolder per patient)")
    parser.add_argument("--luna-dir", type=Path, help="LUNA16 root folder (contains subset*/ with .mhd files)")
    parser.add_argument("--luna-annotations", type=Path, help="LUNA16 annotations.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slices-per-patient", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(args.osic_dir, args.luna_dir, args.luna_annotations,
                  args.output_dir, args.slices_per_patient, args.seed)


if __name__ == "__main__":
    main()
