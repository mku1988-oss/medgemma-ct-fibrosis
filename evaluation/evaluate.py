"""
Patient-level evaluation with 7-slice majority voting.

For each test patient, 7 evenly spaced lower-lobe slices are classified
with the neutral prompt. The patient is called fibrosis when more than half
of the readable votes say fibrosis.

Slices are copied under random file names and shuffled before inference,
so neither file names nor order can carry label information.

Usage:
    python -m evaluation.evaluate --manifest data/processed/manifest.json \
        --splits-dir data/splits --output-dir results

    python -m evaluation.evaluate ... --base     # base model, no fine-tuning

Research use only. Not for clinical use.
"""

import argparse
import json
import logging
import random
import shutil
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm

from inference.predictor import ADAPTER, BASE_MODEL, FibrosisPredictor
from training.prompt import FIBROSIS_ANSWER, NORMAL_ANSWER

from .free_text_parser import parse_free_text
from .metrics import evenly_spaced, patient_metrics, vote

logger = logging.getLogger(__name__)

SLICES_PER_PATIENT = 7


def evaluate(manifest: Path, splits_dir: Path, output_dir: Path,
             adapter: str = ADAPTER, base_model: str = BASE_MODEL, seed: int = 42) -> Dict:
    """Run the evaluation, write per-slice and per-patient CSVs, return metrics."""
    random.seed(seed)
    test_patients = pd.read_csv(splits_dir / "test_split.csv")
    labels = dict(zip(test_patients["patient_id"].astype(str), test_patients["label"]))

    predictor = FibrosisPredictor(adapter=adapter, base_model=base_model)
    use_free_text_parser = adapter is None

    with tempfile.TemporaryDirectory() as blind_dir:
        items = blinded_slices(manifest, labels, Path(blind_dir))
        slice_rows = [classify(predictor, item, use_free_text_parser)
                      for item in tqdm(items, desc="Slices")]

    patient_rows = patient_votes(slice_rows, labels)
    metrics = patient_metrics(patient_rows)
    save_results(slice_rows, patient_rows, metrics, output_dir, "base" if adapter is None else "fine_tuned")
    return metrics


def blinded_slices(manifest: Path, labels: Dict[str, int], blind_dir: Path) -> List[Dict]:
    """Copy 7 evenly spaced slices per test patient under random names, shuffled."""
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    slices_by_patient = defaultdict(list)
    for entry in entries:
        if str(entry["patient_id"]) in labels:
            slices_by_patient[str(entry["patient_id"])].append(entry["image"])

    items = []
    for patient_id, images in slices_by_patient.items():
        for image in evenly_spaced(images, SLICES_PER_PATIENT):
            blind_path = blind_dir / f"{uuid.uuid4().hex}.png"
            shutil.copy2(manifest.parent / image, blind_path)
            items.append({"patient_id": patient_id, "image": blind_path})

    random.shuffle(items)
    return items


def classify(predictor: FibrosisPredictor, item: Dict, use_free_text_parser: bool) -> Dict:
    """Classify one slice. The vote is 1 (fibrosis), 0 (normal) or -1 (unreadable)."""
    prediction = predictor.predict(Image.open(item["image"]))
    if use_free_text_parser:
        slice_vote = parse_free_text(prediction.raw_response)
    else:
        slice_vote = {FIBROSIS_ANSWER: 1, NORMAL_ANSWER: 0}.get(prediction.answer, -1)
    return {"patient_id": item["patient_id"], "vote": slice_vote,
            "confidence": round(prediction.confidence, 4), "response": prediction.raw_response[:200]}


def patient_votes(slice_rows: List[Dict], labels: Dict[str, int]) -> List[Dict]:
    """Combine slice votes into one prediction per patient."""
    votes = defaultdict(list)
    for row in slice_rows:
        votes[row["patient_id"]].append(row["vote"])
    return [{"patient_id": pid, "label": int(labels[pid]), "prediction": vote(v),
             "fibrosis_votes": v.count(1), "normal_votes": v.count(0), "unreadable_votes": v.count(-1)}
            for pid, v in sorted(votes.items())]


def save_results(slice_rows, patient_rows, metrics, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(slice_rows).to_csv(output_dir / f"{name}_slices.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(output_dir / f"{name}_patients.csv", index=False)
    (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Results for %s: %s", name, json.dumps(metrics))


def main():
    parser = argparse.ArgumentParser(description="Patient-level evaluation with 7-slice voting.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--adapter", default=ADAPTER, help="LoRA adapter (Hugging Face repo or local folder)")
    parser.add_argument("--base-model", default=BASE_MODEL, help="4-bit base model (Hugging Face repo or local folder)")
    parser.add_argument("--base", action="store_true", help="Evaluate the base model without the adapter")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate(args.manifest, args.splits_dir, args.output_dir,
             None if args.base else args.adapter, args.base_model)


if __name__ == "__main__":
    main()
