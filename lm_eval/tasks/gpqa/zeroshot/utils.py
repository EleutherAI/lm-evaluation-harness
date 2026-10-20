from __future__ import annotations

import random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets


def _shuffled_choices(choices: list, rng_state: tuple, index: int) -> None:
    """Shuffle as the index-th draw from a captured random state.

    Dataset.map caches the transform and not the global random state. Replaying
    earlier shuffles keeps each row tied to the state from this call, so another
    seed cannot reuse the previous choice order.
    """
    rng = random.Random()
    rng.setstate(rng_state)
    scratch = list(range(len(choices)))
    for _ in range(index):
        rng.shuffle(scratch)
    rng.shuffle(choices)


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")

    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng_state = random.getstate()

    def _process_doc(doc, idx, rng_state):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        _shuffled_choices(choices, rng_state, idx)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(
        _process_doc, with_indices=True, fn_kwargs={"rng_state": rng_state}
    )
