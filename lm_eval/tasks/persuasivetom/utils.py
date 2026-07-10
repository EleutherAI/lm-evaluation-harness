import json
import re
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
    return {
        **doc,
        "choices": choices,
        "choice_letters": list(LETTERS[: len(choices)]),
        "answerKey": str(doc.get("answerKey", "")),
    }


def load(data_file=None, **kwargs):
    if not data_file:
        raise ValueError("PersuasiveToM leaf tasks must pass dataset_kwargs.data_file")
    path = _data_dir() / data_file
    if not path.exists():
        raise FileNotFoundError(f"PersuasiveToM data file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed = [_parse_doc(row) for row in rows]
    return {"test": datasets.Dataset.from_list(parsed)}


def doc_to_text(doc):
    options = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(doc["choice_letters"], doc["choices"])
    )
    return (
        f"\nDialogue History:
{doc['dialogue']}
"
        f"Question:
{doc['question']}
"
        f"Choices:
{options}
"
        "Answer:"
    )


def _parse_prediction(text, valid_letters):
    candidate = text.strip()
    for prefix in ("Answer:", "answer:", "The answer is", "the answer is"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
            break
    candidate = (
        candidate.replace('"', "")
        .replace("*", "")
        .replace(":", "")
        .replace(".", "")
        .replace("-", "")
        .replace(",", "")
        .split("\n")[0]
        .strip()
    )
    if candidate[:1] in valid_letters:
        return candidate[:1]
    match = re.search(r"\b([A-F])\b", candidate)
    if match and match.group(1) in valid_letters:
        return match.group(1)
    return candidate.split()[0] if candidate.split() else "Z"


def process_results(doc, results):
    pred = _parse_prediction(results[0], set(doc["choice_letters"]))
    return {"acc": 1.0 if pred == doc["answerKey"] else 0.0}
