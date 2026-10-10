"""Helpers for the PT-BR typed-decisions tasks in lm-evaluation-harness.

Each item is a closed question about a Brazilian Portuguese text. Options are listed with letters and the model
scores each letter (loglikelihood). Balanced accuracy (mean per-class recall) is the headline metric, as on the card.
"""

from collections import defaultdict


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_CHARS = 3500


def _options(doc):
    if doc["type"] == "noul":
        return [("nao", "Não"), ("sim", "Sim")]
    return [(k, v) for k, v in doc["criteria"].items() if v is not None]


def doc_to_text(doc):
    opts = "\n".join(
        f"{LETTERS[i]}) {desc}" for i, (_, desc) in enumerate(_options(doc))
    )
    return f"Texto:\n{doc['text'][:MAX_CHARS]}\n\nPergunta: {doc['instructions']}\nOpções:\n{opts}\nResposta:"


def doc_to_choice(doc):
    return [f" {LETTERS[i]}" for i in range(len(_options(doc)))]


def doc_to_target(doc):
    keys = [k for k, _ in _options(doc)]
    gold = ("sim" if doc["gold"] else "nao") if doc["type"] == "noul" else doc["gold"]
    return keys.index(gold)


def process_results(doc, results):
    lls = [r[0] for r in results]
    pred = max(range(len(lls)), key=lls.__getitem__)
    gold = doc_to_target(doc)
    return {"acc": float(pred == gold), "balanced_acc": (gold, pred)}


def balanced_acc(items):
    by = defaultdict(lambda: [0, 0])
    for gold, pred in items:
        by[gold][0] += gold == pred
        by[gold][1] += 1
    return sum(c / n for c, n in by.values()) / len(by)
