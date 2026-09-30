"""Scoring helpers for the canine_geroscience tasks.

Gold answers in the dataset are short free-text spans (a number with units, a phrase, or a
short list), so generation is scored with:

* ``exact_match``: normalised string equality (lower-case, punctuation and articles removed).
* ``f1``: SQuAD-style token-overlap F1 between the normalised prediction and gold answer.
* ``numeric_acc`` (numeric subset only): 1.0 when the gold answer's primary figure (its first
  number outside parentheses, which hold confidence intervals) appears in the prediction
  (commas removed, compared as floats), else 0.0.
* ``numeric_acc_all`` (numeric subset only): the stricter variant requiring every number
  outside parentheses, for multi-part answers such as a dose and a frequency.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import datasets

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
# A number: optional sign (not when glued to a preceding digit, so "11.19-11.27" is two
# positive numbers), digits with thousands separators, optional decimals.
_NUMBER = re.compile(r"(?<![\d.])-?\d[\d,]*\.?\d*")
# Parentheticals that report a spread rather than the answer, e.g. "(95% CI 12.68-12.70)".
_SPREAD_PARENTHETICAL = re.compile(
    r"\([^)]*(?:\bCI\b|confidence|\bSD\b|\bSE\b|\bSEM\b|\bIQR\b|±|standard deviation)[^)]*\)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text):
        s = m.replace(",", "").rstrip(".")
        if not s or s == "-":
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def _present(value: float, candidates: list[float]) -> bool:
    return any(abs(value - c) <= 1e-6 * max(1.0, abs(value)) for c in candidates)


def gold_numbers(gold: str) -> list[float]:
    """Numbers in the gold answer, ignoring parentheticals that report a spread
    (confidence interval, SD, IQR), e.g. "12.69 years (95% CI 12.68-12.70)".
    """
    return numbers_in(_SPREAD_PARENTHETICAL.sub(" ", gold))


def numeric_match(pred: str, gold: str) -> float:
    """1.0 when the gold answer's primary figure (its first number, ignoring spread
    parentheticals) appears in the prediction.
    """
    nums = gold_numbers(gold)
    if not nums:
        return float(normalize(gold) in normalize(pred))
    return float(_present(nums[0], numbers_in(pred)))


def numeric_match_all(pred: str, gold: str) -> float:
    """1.0 when every number in the gold answer (ignoring spread parentheticals) appears
    in the prediction (stricter: multi-part answers such as a dose and a frequency).
    """
    nums = gold_numbers(gold)
    if not nums:
        return float(normalize(gold) in normalize(pred))
    pred_nums = numbers_in(pred)
    return float(all(_present(g, pred_nums) for g in nums))


def process_docs_numeric(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda doc: doc["answer_type"] == "numeric")


def _prediction(results) -> str:
    pred = results[0] if isinstance(results, (list, tuple)) else results
    return (pred or "").strip()


def process_results(doc: dict, results) -> dict:
    pred, gold = _prediction(results), doc["answer"]
    return {
        "exact_match": float(normalize(pred) == normalize(gold)),
        "f1": token_f1(pred, gold),
    }


def process_results_numeric(doc: dict, results) -> dict:
    pred, gold = _prediction(results), doc["answer"]
    return {
        "numeric_acc": numeric_match(pred, gold),
        "numeric_acc_all": numeric_match_all(pred, gold),
        "exact_match": float(normalize(pred) == normalize(gold)),
    }


def list_fewshot_samples() -> list[dict]:
    """Three hand-written examples in the dataset's format.

    None of them is an item of the evaluation set; they only show the expected answer style.
    """
    return [
        {
            "question": "How many dogs were enrolled in Loyal's pivotal STAY trial of LOY-002, "
            "which completed enrollment in 2025?",
            "answer": "1,300 dogs.",
        },
        {
            "question": "What does the acronym CCDR stand for in canine cognition research?",
            "answer": "The Canine Cognitive Dysfunction Rating scale.",
        },
        {
            "question": "Which two FDA-accepted technical sections of the conditional approval "
            "application for LOY-002 had been completed by January 2026?",
            "answer": "Reasonable expectation of effectiveness (RXE) and target animal safety (TAS).",
        },
    ]
