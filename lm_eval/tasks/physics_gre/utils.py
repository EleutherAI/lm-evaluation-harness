from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert BIG-bench-style ``target_scores`` into choices/gold fields.

    The Physics GRE items store the answer as a mapping such as
    ``{"A": 0, "B": 1, "C": 0, "D": 0, "E": 0}`` where the correct option has a
    score of 1. Following the original Inflection-Benchmarks methodology, only
    image-free questions are scored, so items with ``has_image`` are dropped.
    """

    def _is_scorable(doc: dict) -> bool:
        # Score only image-free questions, matching the original methodology.
        if doc["has_image"]:
            return False
        # A few source items have a missing answer key (``target_scores`` null
        # or no option scored 1); these cannot be graded, so drop them.
        scores = doc["target_scores"]
        return scores is not None and 1 in scores.values()

    def _process_doc(doc: dict) -> dict:
        target_scores = doc["target_scores"]
        choices = list(target_scores.keys())
        gold = list(target_scores.values()).index(1)
        return {
            "input": doc["input"].strip(),
            "choices": choices,
            "gold": gold,
        }

    return dataset.filter(_is_scorable).map(_process_doc)
