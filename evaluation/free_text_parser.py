"""
Read a yes/no verdict from the base model's free-text answers.

The fine-tuned model replies with a fixed phrase. The base model, without
fine-tuning, writes a paragraph of reasoning instead, so a stricter parser
would count almost all of its answers as unreadable (0% in our tests). This
lenient parser is used only for the base-model comparison, so that the base
model is judged on what it says rather than on its format.

Returns 1 (fibrosis), 0 (normal) or -1 (no clear verdict).
"""

import re

_FINAL_YES = re.compile(r"(?:final\s+answer|my\s+answer\s+is)[:\s]*(?:\*{0,2}|[$\\]+boxed\{)\s*(?:yes|fibrosis)")
_FINAL_NO = re.compile(r"(?:final\s+answer|my\s+answer\s+is)[:\s]*(?:\*{0,2}|[$\\]+boxed\{)\s*(?:no\b|normal)")

_CONCLUSION_WORDS = ["therefore", "conclusion", "in summary", "based on this", "based on the provided",
                     "based on the image", "overall", "in conclusion"]

_NEGATIVE = [
    r"no\s+\w*\s*(?:evidence|signs?|indication)\s+(?:of\s+)?(?:pulmonary\s+)?fibrosis",
    r"there\s+is\s+(?:\*{0,2})?\s*no\s+\w*\s*(?:\*{0,2})?\s*(?:evidence|signs?)",
    r"does\s+not\s+(?:show|exhibit|suggest|indicate)",
    r"i\s+do\s+not\s+see",
    r"(?:my\s+)?answer\s+is[:\s]*\*{0,2}\s*no\b",
]
_POSITIVE = [
    r"there\s+is\s+(?:\*{0,2})?\s*(?:some\s+)?evidence\s+(?:of\s+)?(?:\*{0,2})?\s*(?:pulmonary\s+)?fibrosis",
    r"(?:suggestive|consistent|indicative)\s+(?:of|with)\s+(?:pulmonary\s+)?fibrosis",
    r"findings?\s+(?:strongly\s+)?suggest(?:s|ing)?\s+(?:pulmonary\s+)?fibrosis",
    r"\byes\b,?\s+there\s+is\s+evidence",
    r"(?:my\s+)?answer\s+is[:\s]*\*{0,2}\s*yes\b",
]


def parse_free_text(text: str) -> int:
    """Return 1, 0 or -1 for a free-text answer."""
    lower = text.lower()

    # 1. An explicit "final answer" wins if it is unambiguous.
    said_yes, said_no = bool(_FINAL_YES.search(lower)), bool(_FINAL_NO.search(lower))
    if said_yes != said_no:
        return 1 if said_yes else 0

    # 2. Otherwise score the sentences most likely to hold the verdict.
    positive = negative = 0
    for sentence in _verdict_sentences(text):
        s = sentence.lower()
        if any(re.search(p, s) for p in _NEGATIVE):
            negative += 1
        elif any(re.search(p, s) for p in _POSITIVE):
            positive += 1

    if positive > negative:
        return 1
    if negative > positive:
        return 0
    return -1


def _verdict_sentences(text: str):
    """The first sentence, any concluding sentences, and the last sentence."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip()) if s.strip()]
    if not sentences:
        return []

    chosen = [sentences[0]]
    chosen += [s for s in sentences if any(w in s.lower() for w in _CONCLUSION_WORDS) and s not in chosen]
    if sentences[-1] not in chosen:
        chosen.append(sentences[-1])
    return chosen
