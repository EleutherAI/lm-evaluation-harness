import json
from pathlib import Path

import datasets


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _data_dir():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks/PersuasiveToM/data"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "PersuasiveToM data directory not found under any parent's benchmarks/ directory. "
        "Run `git submodule update --init --recursive benchmarks/PersuasiveToM` from "
        "the tomverse-of-madness repo root."
    )


def _parse_doc(doc):
    choices = list(doc["choices"])
    answer = str(doc.get("answerKey", "")).strip().upper()
    valid_letters = LETTERS[: len(choices)]
    if answer not in valid_letters:
        return None
    return {
        **doc,
        "choices": choices,
        "choice_letters": list(valid_letters),
        "gold": valid_letters.index(answer),
    }


def load(data_file=None, **kwargs):
    if not data_file:
        raise ValueError("PersuasiveToM leaf tasks must pass dataset_kwargs.data_file")
    path = _data_dir() / data_file
    if not path.exists():
        raise FileNotFoundError(f"PersuasiveToM data file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed = [_parse_doc(row) for row in rows]
    parsed = [row for row in parsed if row is not None]
    return {"test": datasets.Dataset.from_list(parsed)}


def doc_to_text(doc):
    options = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(doc["choice_letters"], doc["choices"])
    )
    return (
        "Here is a persuasive dialogue. Read the dialogue history and answer the "
        "multiple-choice question.\n"
        f"Dialogue History:\n{doc['dialogue']}\n"
        f"Question:\n{doc['question']}\n"
        f"Choices:\n{options}\n"
        "Answer:"
    )
