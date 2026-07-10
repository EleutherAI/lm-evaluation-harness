import json
from pathlib import Path

import datasets


LABELS = ["[A]", "[B]", "[C]", "[D]"]


def _dataset_dir():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks/ToMATO/dataset"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "ToMATO dataset directory not found under any parent's benchmarks/ directory. "
        "Run `git submodule update --init --recursive benchmarks/ToMATO` from the "
        "tomverse-of-madness repo root."
    )


def _parse_doc(doc):
    transcript = doc.get("transcript", doc.get("conversation"))
    if transcript is None:
        raise ValueError(f"ToMATO row is missing both transcript and conversation: {doc}")
    gold_index = int(doc["a_idx"])
    if gold_index not in range(len(LABELS)):
        raise ValueError(f"ToMATO row has invalid a_idx={gold_index}: {doc}")
    return {
        **doc,
        "transcript": transcript,
        "choice_labels": list(LABELS),
        "gold_label": LABELS[gold_index],
    }


def load(data_file="tomato.json", **kwargs):
    path = _dataset_dir() / data_file
    if not path.exists():
        raise FileNotFoundError(f"ToMATO data file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {"test": datasets.Dataset.from_list([_parse_doc(row) for row in rows])}


def doc_to_text(doc):
    return (
        f"# Transcript \n{doc['transcript']}\n\n"
        f"# Question \n{doc['q']}\n\n"
        "# Options \n"
        f"[A] {doc['a0']}\n"
        f"[B] {doc['a1']}\n"
        f"[C] {doc['a2']}\n"
        f"[D] {doc['a3']}\n"
    )


def process_docs_order_1(dataset):
    return dataset.filter(lambda doc: int(doc["order"]) == 1)


def process_docs_order_2(dataset):
    return dataset.filter(lambda doc: int(doc["order"]) == 2)


def process_docs_mental_state_belief(dataset):
    return dataset.filter(lambda doc: doc["mental_state"] == "belief")


def process_docs_mental_state_desire(dataset):
    return dataset.filter(lambda doc: doc["mental_state"] == "desire")


def process_docs_mental_state_emotion(dataset):
    return dataset.filter(lambda doc: doc["mental_state"] == "emotion")


def process_docs_mental_state_intention(dataset):
    return dataset.filter(lambda doc: doc["mental_state"] == "intention")


def process_docs_mental_state_knowledge(dataset):
    return dataset.filter(lambda doc: doc["mental_state"] == "knowledge")


def process_docs_false_belief_true(dataset):
    return dataset.filter(lambda doc: bool(doc["false_belief"]) is True)


def process_docs_false_belief_false(dataset):
    return dataset.filter(lambda doc: bool(doc["false_belief"]) is False)
