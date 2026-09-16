"""Scoring for the LooGLE long-context generation tasks.

LooGLE: Can Long-Context Language Models Understand Long Contexts?
Jiaqi Li et al., ACL 2024 — https://arxiv.org/abs/2311.04939
Benchmark: https://github.com/bigai-nlco/LooGLE (MIT)
Data:      https://huggingface.co/datasets/bigai-nlco/LooGLE (MIT)

Covers the three free-text generation sub-tasks (shortdep_qa, longdep_qa,
summarization), scored with ROUGE. The cloze sub-task (entity fill-in, scored
by exact/partial entity match) is left for a follow-up.

ROUGE is re-implemented here in the standard library (LCS for ROUGE-L, n-gram
overlap for ROUGE-1/2) so the task needs no extra dependencies. F-measure is
reported; the LooGLE paper additionally reports the recall variant.
"""

from __future__ import annotations

import string
from collections import Counter


_PUNCT = str.maketrans("", "", string.punctuation)


def _tokens(text: str) -> list[str]:
    """Lowercase, drop punctuation, split on whitespace."""
    return text.lower().translate(_PUNCT).split()


def _f_measure(overlap: int, pred_total: int, gold_total: int) -> float:
    if overlap == 0 or pred_total == 0 or gold_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / gold_total
    return 2 * precision * recall / (precision + recall)


def _rouge_n(pred: list[str], gold: list[str], n: int) -> float:
    if len(pred) < n or len(gold) < n:
        return 0.0
    pred_ngrams = Counter(tuple(pred[i : i + n]) for i in range(len(pred) - n + 1))
    gold_ngrams = Counter(tuple(gold[i : i + n]) for i in range(len(gold) - n + 1))
    overlap = sum((pred_ngrams & gold_ngrams).values())
    return _f_measure(overlap, sum(pred_ngrams.values()), sum(gold_ngrams.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _rouge_l(pred: list[str], gold: list[str]) -> float:
    if not pred or not gold:
        return 0.0
    lcs = _lcs_length(pred, gold)
    return _f_measure(lcs, len(pred), len(gold))


def process_results(doc: dict, results: list[str]) -> dict:
    prediction = _tokens(results[0])
    reference = _tokens(doc["answer"])
    rouge_l = _rouge_l(prediction, reference)
    return {
        "rouge_l": rouge_l,
        "rouge_1": _rouge_n(prediction, reference, 1),
        "rouge_2": _rouge_n(prediction, reference, 2),
    }
