import random
import re

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Prepare GPQA docs for AAII zero-shot evaluation.

    This mirrors the AAII approach by creating shuffled choices and a
    single-letter target formatted as "A"/"B" etc. The prompt text is
    constructed in the YAML `doc_to_text` include.
    """

    def _process_doc(doc):
        choices = [
            preprocess(doc.get("Incorrect Answer 1")),
            preprocess(doc.get("Incorrect Answer 2")),
            preprocess(doc.get("Incorrect Answer 3")),
            preprocess(doc.get("Correct Answer")),
        ]

        # shuffle choices deterministically per-run randomness is OK here
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc.get("Correct Answer")))

        out_doc = {
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    return dataset.map(_process_doc)
