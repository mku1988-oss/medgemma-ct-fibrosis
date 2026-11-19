"""
Split patients into training, validation and test sets.

The split is made at patient level, so all slices of one patient end up in
the same set. It is stratified by label and uses a 64 / 16 / 20 ratio:
20% of patients are held out for testing, and 20% of the remainder for
validation.

Usage:
    python -m preprocessing.make_splits --manifest data/processed/manifest.json \
        --output-dir data/splits
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def patient_table(manifest_path: Path) -> pd.DataFrame:
    """One row per patient with its label. Raises if a patient has mixed labels."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    table = pd.DataFrame(manifest)[["patient_id", "label"]].drop_duplicates()

    mixed = table["patient_id"][table["patient_id"].duplicated()]
    if not mixed.empty:
        raise ValueError(f"Patients with more than one label: {sorted(mixed.unique())}")

    return table.reset_index(drop=True)


def make_splits(patients: pd.DataFrame,
                test_fraction: float = 0.20,
                val_fraction: float = 0.20,
                seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"} patient tables."""
    train_val, test = train_test_split(
        patients, test_size=test_fraction, stratify=patients["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["label"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def save_splits(splits: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in splits.items():
        table.to_csv(output_dir / f"{name}_split.csv", index=False)
        logger.info("%-5s %3d patients (%d fibrosis)", name, len(table), int(table["label"].sum()))


def main():
    parser = argparse.ArgumentParser(description="Create patient-level stratified splits.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    save_splits(make_splits(patient_table(args.manifest), seed=args.seed), args.output_dir)


if __name__ == "__main__":
    main()
