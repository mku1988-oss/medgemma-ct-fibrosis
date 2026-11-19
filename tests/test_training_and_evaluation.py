"""
Tests for the prompt format, dataset sampling, splits, voting and metrics.

None of these tests load the model, so they run without a GPU.
"""

import json
import random

import pandas as pd
import pytest

from evaluation.free_text_parser import parse_free_text
from evaluation.metrics import evenly_spaced, patient_metrics, vote
from preprocessing.make_splits import make_splits, patient_table
from training.dataset import SliceDataset
from training.prompt import (FIBROSIS_ANSWER, NEUTRAL_CONTEXT, NORMAL_ANSWER, UNPARSEABLE,
                             answer_for_label, build_messages, parse_answer)


@pytest.fixture
def manifest(tmp_path):
    """A manifest with 2 fibrosis and 3 normal patients, 3 slices each."""
    entries = [{"patient_id": f"{kind}{p}", "label": label, "image": f"images/{kind}{p}_slice_{s}.png"}
               for kind, label, count in [("fib", 1, 2), ("nor", 0, 3)]
               for p in range(count) for s in range(3)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries))
    return path


# ---------------------------------------------------------------------------
# Prompt and answers
# ---------------------------------------------------------------------------

def test_messages_put_image_before_text():
    content = build_messages("slice.png")[0]["content"]
    assert content[0] == {"type": "image", "image": "slice.png"}
    assert content[1]["text"].startswith(NEUTRAL_CONTEXT)


@pytest.mark.parametrize("text, expected", [
    ("NO FIBROSIS DETECTED", NORMAL_ANSWER),
    ("fibrosis detected", FIBROSIS_ANSWER),
    ("I am not sure.", UNPARSEABLE),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_answers_round_trip_through_parser():
    assert parse_answer(answer_for_label(1)) == FIBROSIS_ANSWER
    assert parse_answer(answer_for_label(0)) == NORMAL_ANSWER


@pytest.mark.parametrize("text, expected", [
    ("There is evidence of pulmonary fibrosis in the lower lobes.", 1),
    ("There is no evidence of pulmonary fibrosis.", 0),
    ("Final answer: **Yes**", 1),
    ("The image quality is limited.", -1),
])
def test_free_text_parser(text, expected):
    assert parse_free_text(text) == expected


# ---------------------------------------------------------------------------
# Dataset and splits
# ---------------------------------------------------------------------------

def test_dataset_oversamples_fibrosis(manifest):
    dataset = SliceDataset(manifest, fibrosis_oversampling=2)
    assert len(dataset) == 3 + 2 * 2


def test_dataset_respects_patient_filter(manifest):
    dataset = SliceDataset(manifest, patient_ids=["fib0", "nor0"])
    assert sorted(dataset.label_by_patient) == ["fib0", "nor0"]


def test_dataset_item_is_one_slice_of_the_right_patient(manifest):
    random.seed(0)
    dataset = SliceDataset(manifest, patient_ids=["fib1"])
    messages = dataset[0]["messages"]

    image_path = messages[0]["content"][0]["image"]
    assert "fib1_slice_" in image_path
    assert messages[1]["content"][0]["text"] == FIBROSIS_ANSWER


def test_splits_are_patient_level_and_stratified(tmp_path):
    entries = [{"patient_id": f"p{i}", "label": int(i < 40), "image": f"images/p{i}.png"} for i in range(100)]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(entries))

    splits = make_splits(patient_table(path))
    sets = [set(s["patient_id"]) for s in splits.values()]

    assert sum(len(s) for s in sets) == 100
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert splits["test"]["label"].mean() == pytest.approx(0.4, abs=0.05)


def test_mixed_labels_are_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([{"patient_id": "p", "label": 1, "image": "a"},
                                {"patient_id": "p", "label": 0, "image": "b"}]))
    with pytest.raises(ValueError):
        patient_table(path)


# ---------------------------------------------------------------------------
# Voting and metrics
# ---------------------------------------------------------------------------

def test_evenly_spaced_includes_both_ends():
    assert evenly_spaced(list(range(13)), 7) == [0, 2, 4, 6, 8, 10, 12]
    assert evenly_spaced([1, 2], 7) == [1, 2]


@pytest.mark.parametrize("votes, expected", [
    ([1, 1, 1, 1, 0, 0, 0], 1),
    ([1, 1, 1, 0, 0, 0, 0], 0),
    ([1, 1, 0, 0], 0),          # a tie counts as normal
    ([1, 1, -1, -1, -1], 1),    # unreadable votes are ignored
    ([-1, -1], 0),
])
def test_majority_vote(votes, expected):
    assert vote(votes) == expected


def test_patient_metrics_match_released_model_counts():
    # Neutral-prompt confusion matrix of the released model: 29/32 fibrosis, 43/49 normal.
    rows = ([{"label": 1, "prediction": 1}] * 29 + [{"label": 1, "prediction": 0}] * 3 +
            [{"label": 0, "prediction": 0}] * 43 + [{"label": 0, "prediction": 1}] * 6)
    metrics = patient_metrics(rows)

    assert metrics["accuracy"] == pytest.approx(0.8889, abs=1e-4)
    assert metrics["sensitivity"] == pytest.approx(0.9062, abs=1e-4)
    assert metrics["specificity"] == pytest.approx(0.8776, abs=1e-4)
