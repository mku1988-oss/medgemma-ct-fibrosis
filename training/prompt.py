"""
Prompt and answer format shared by training, evaluation and the demo.

The model must see exactly the same prompt at inference time as during
training, so every part of the project imports it from here.
"""

from typing import List

# Neutral clinical context. It carries no information about the label, so
# the model has to base its answer on the image.
NEUTRAL_CONTEXT = "Patient referred for chest CT evaluation. No additional clinical history available."

QUESTION = (
    "Examine this CT slice from the lower lung region for signs of pulmonary fibrosis "
    "(honeycombing, reticulation, traction bronchiectasis, ground-glass opacity).\n"
    "Is there evidence of pulmonary fibrosis in this image?"
)

FIBROSIS_ANSWER = "FIBROSIS DETECTED"
NORMAL_ANSWER = "NO FIBROSIS DETECTED"
UNPARSEABLE = "UNPARSEABLE"


def user_text(context: str = NEUTRAL_CONTEXT) -> str:
    """The text part of the user turn: clinical context, blank line, question."""
    return f"{context}\n\n{QUESTION}"


def build_messages(image, context: str = NEUTRAL_CONTEXT) -> List[dict]:
    """
    Chat messages for one slice, in the format MedGemma's processor expects.

    Args:
        image: A PIL image or a path to an image file.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text(context)},
        ],
    }]


def answer_for_label(label: int) -> str:
    """Training target for a label (1 = fibrosis, 0 = normal)."""
    return FIBROSIS_ANSWER if label == 1 else NORMAL_ANSWER


def parse_answer(text: str) -> str:
    """
    Map a generated response to FIBROSIS_ANSWER, NORMAL_ANSWER or UNPARSEABLE.

    The normal answer contains the fibrosis answer as a substring, so it is
    checked first.
    """
    upper = text.upper()
    if NORMAL_ANSWER in upper:
        return NORMAL_ANSWER
    if FIBROSIS_ANSWER in upper:
        return FIBROSIS_ANSWER
    return UNPARSEABLE
