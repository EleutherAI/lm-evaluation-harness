from __future__ import annotations

import json
import os
import random
from functools import partial
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


_HERE = os.path.dirname(__file__)


def _load_exclusions(name):
    # Record IDs removed by the answer-key audit (Allcock 2026): wrong key,
    # malformed item, or more than one defensible answer. Shipped as GPQA
    # `Record ID`s so the filter needs no id remapping.
    with open(os.path.join(_HERE, name)) as f:
        return set(json.load(f))


def _process(dataset: "datasets.Dataset", exclusions: set) -> "datasets.Dataset":
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))
        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    # Corrected variant: drop audit-flagged broken items, then apply the same
    # choice-shuffling processing as the upstream `gpqa` task.
    return dataset.filter(lambda d: d["Record ID"] not in exclusions).map(_process_doc)


process_diamond_clean = partial(
    _process, exclusions=_load_exclusions("gpqa_diamond_clean_exclusions.json")
)
process_extended_clean = partial(
    _process, exclusions=_load_exclusions("gpqa_extended_clean_exclusions.json")
)
