"""ARC-Challenge with all choices visible in every possible order."""

from collections import defaultdict
from itertools import permutations

from datasets import Dataset


def process_docs(dataset):
    rows = []
    for row in dataset:
        choices = row["choices"]
        gold = choices["label"].index(row["answerKey"])
        for order in permutations(range(len(choices["text"]))):
            rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "choices": {"text": [choices["text"][i] for i in order]},
                    "gold": order.index(gold),
                    "order": order,
                }
            )
    return Dataset.from_list(rows)


def doc_to_text(doc):
    return (
        f"Question: {doc['question']}\nChoices:\n"
        + "\n".join(doc["choices"]["text"])
        + "\nAnswer:"
    )


def process_results(doc, results):
    scores = [result[0] for result in results]
    gold = doc["gold"]
    texts = doc["choices"]["text"]
    correct = int(max(range(len(texts)), key=scores.__getitem__) == gold)
    correct_norm = int(
        max(range(len(texts)), key=lambda i: scores[i] / len(texts[i])) == gold
    )
    item = (doc["id"], correct)
    item_norm = (doc["id"], correct_norm)
    return {
        "acc": item,
        "acc_adv": item,
        "acc_norm": item_norm,
        "acc_norm_adv": item_norm,
    }


def mean_accuracy(items):
    by_question = defaultdict(list)
    for question_id, correct in items:
        by_question[question_id].append(correct)
    return sum(sum(values) / len(values) for values in by_question.values()) / len(
        by_question
    )


def worst_accuracy(items):
    worst = {}
    for question_id, correct in items:
        worst[question_id] = min(worst.get(question_id, 1), correct)
    return sum(worst.values()) / len(worst)


if __name__ == "__main__":
    source = Dataset.from_list(
        [
            {
                "id": "example",
                "question": "Which is smallest?",
                "choices": {
                    "text": ["four", "one", "three", "two"],
                    "label": ["1", "2", "3", "4"],
                },
                "answerKey": "2",
            },
            {
                "id": "three",
                "question": "Three choices",
                "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]},
                "answerKey": "A",
            },
        ]
    )
    docs = process_docs(source)
    assert len(docs) == 24 + 6
    assert all(
        doc["choices"]["text"][doc["gold"]] == "one"
        for doc in docs
        if doc["id"] == "example"
    )
    assert len({tuple(doc["order"]) for doc in docs if doc["id"] == "example"}) == 24
    assert len({tuple(doc["order"]) for doc in docs if doc["id"] == "three"}) == 6
    assert (
        doc_to_text(docs[0])
        == "Question: Which is smallest?\nChoices:\nfour\none\nthree\ntwo\nAnswer:"
    )
    assert (
        process_results(docs[0], [(-4, False), (-1, False), (-3, False), (-2, False)])[
            "acc"
        ][1]
        == 1
    )
    assert mean_accuracy([("a", 1), ("a", 0), ("b", 1)]) == 0.75
    assert mean_accuracy([("three", 1)] * 6 + [("five", 0)] * 120) == 0.5
    assert worst_accuracy([("a", 1), ("a", 0), ("b", 1)]) == 0.5
