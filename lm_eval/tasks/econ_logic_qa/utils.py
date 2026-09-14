"""Utilities for the EconLogicQA task.

EconLogicQA asks a model to order four interconnected economic events by
*logical* rather than merely chronological precedence.  The gold answer is a
permutation of the option letters, e.g. ``"D, A, C, B"``.

The paper (Quan & Liu, 2024, arXiv:2405.07938) reports accuracy after
extracting the permutation from the generation with a regular expression and
comparing it to the gold order with exact matching; ``exact_match`` below
reproduces that.  ``pairwise_accuracy`` is an additional diagnostic described
in the task README.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any


LETTERS = ("A", "B", "C", "D")

# The separated form models actually emit: "D, A, C, B", "D -> A -> C -> B",
# "D > A > C > B", "D A C B".  A bare "." is deliberately *not* a separator, so
# an enumerated restatement of the options ("A. The companies identify ...")
# cannot be mistaken for an ordering.
_SEPARATOR = r"\s*(?:,|;|->|-->|→|>|\||\band\b|\bthen\b)?\s*"
_RUN_SEPARATED = re.compile(
    rf"(?<![A-Za-z])[ABCD](?:{_SEPARATOR}(?<![A-Za-z])[ABCD](?![A-Za-z])){{3}}",
    re.IGNORECASE,
)
# The unseparated form: "DACB".  Upper case only -- lower-case "dacb" is not a
# form models emit, and matching it would swallow ordinary four-letter words.
_RUN_TIGHT = re.compile(r"(?<![A-Za-z])[ABCD]{4}(?![A-Za-z])")
# A standalone capital option letter, for the last-resort scan.  Restricted to
# upper case there because a lower-case "a" is usually the English article.
_STANDALONE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")
# "Answer:", "the correct order is", "final sequence -" ...
_ANSWER_MARKER = re.compile(
    r"(?i)\b(?:answer|order|sequence|ordering)\b\s*(?:is)?\s*[:\-–]?\s*"
)


def doc_to_text(doc: dict[str, Any]) -> str:
    """Render one document as a completion-style prompt."""
    options = "\n".join(f"{letter}. {doc[letter]}" for letter in LETTERS)
    return f"Question: {doc['Question']}\n{options}\nAnswer:"


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    gold = parse_order(doc["Answer"])
    pred = extract_order(results[0] if results else "")
    return {
        "exact_match": float(pred == gold),
        "pairwise_accuracy": pairwise_accuracy(pred, gold),
    }


def parse_order(answer: str) -> list[str]:
    """Read a gold answer such as ``"D, A, C, B"`` into ``["D", "A", "C", "B"]``."""
    letters = re.findall(r"[ABCD]", str(answer).upper())
    return letters if _is_permutation(letters) else []


def extract_order(text: str) -> list[str]:
    """Pull the predicted permutation out of a model generation.

    Returns ``[]`` when the generation contains no permutation of A-D, which
    scores zero on both metrics.
    """
    if not text:
        return []

    # Prefer whatever follows the *last* answer marker: a model that reasons
    # first will mention several partial orderings before committing to one.
    markers = list(_ANSWER_MARKER.finditer(text))
    segments = [text[markers[-1].end() :], text] if markers else [text]

    for segment in segments:
        for pattern in (_RUN_SEPARATED, _RUN_TIGHT):
            for match in pattern.finditer(segment):
                letters = re.findall(r"[ABCD]", match.group(0).upper())
                if _is_permutation(letters):
                    return letters
        # Last resort: the first window of four standalone capitals that is a
        # permutation, which catches prose like "first D, next A, then C, B".
        standalone = _STANDALONE.findall(segment)
        for i in range(len(standalone) - 3):
            window = standalone[i : i + 4]
            if _is_permutation(window):
                return window
    return []


def pairwise_accuracy(pred: list[str], gold: list[str]) -> float:
    """Fraction of the six event pairs whose relative order the model got right.

    Chance is 0.5 for a random permutation; a generation with no recoverable
    ordering scores 0.0.  This separates "ordered the events wrongly" from
    "never produced an ordering at all", which ``exact_match`` conflates.
    """
    if not _is_permutation(pred) or not _is_permutation(gold):
        return 0.0
    rank = {letter: i for i, letter in enumerate(pred)}
    pairs = list(combinations(gold, 2))
    correct = sum(rank[first] < rank[second] for first, second in pairs)
    return correct / len(pairs)


def _is_permutation(letters: list[str]) -> bool:
    return sorted(letters) == list(LETTERS)
