"""Slice selection, majority voting and patient-level metrics."""

from typing import Dict, List


def evenly_spaced(items: List, n: int) -> List:
    """Pick n items spread evenly from first to last (all of them if there are fewer)."""
    if len(items) <= n:
        return list(items)
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def vote(slice_votes: List[int]) -> int:
    """
    Patient prediction from slice votes (1 fibrosis, 0 normal, -1 unreadable).

    Unreadable votes are ignored. The patient is called fibrosis when more
    than half of the remaining votes say fibrosis; a tie counts as normal.
    """
    readable = [v for v in slice_votes if v != -1]
    if not readable:
        return 0
    return 1 if sum(readable) / len(readable) > 0.5 else 0


def patient_metrics(patient_rows: List[Dict]) -> Dict:
    """Accuracy, sensitivity, specificity, precision and F1 over patients."""
    tp = sum(1 for r in patient_rows if r["label"] == 1 and r["prediction"] == 1)
    tn = sum(1 for r in patient_rows if r["label"] == 0 and r["prediction"] == 0)
    fp = sum(1 for r in patient_rows if r["label"] == 0 and r["prediction"] == 1)
    fn = sum(1 for r in patient_rows if r["label"] == 1 and r["prediction"] == 0)

    sensitivity = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    return {
        "patients": len(patient_rows),
        "accuracy": _ratio(tp + tn, len(patient_rows)),
        "sensitivity": sensitivity,
        "specificity": _ratio(tn, tn + fp),
        "precision": precision,
        "f1": _ratio(2 * precision * sensitivity, precision + sensitivity),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
