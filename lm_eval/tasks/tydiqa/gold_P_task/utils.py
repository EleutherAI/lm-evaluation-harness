"""
Code taken as per the official repo
Mentioned in (from official repo): https://github.com/google-research-datasets/tydiqa/blob/master/gold_passage_baseline/eval_gold_passage_baseline.sh
Mentioned Repo: https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
"""

import re
import string
from collections import Counter

import datasets


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def process_results(
    doc,
    results,
):
    ground_truths = doc["answers"]
    prediction = results[0].strip()
    exact_match = metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths
    )
    f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    return {"exact_match": exact_match, "f1": f1}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "context": doc["context"],
            "question": doc["question"],
            "answers": doc["answers"]["text"][0],
        }
        return out_doc

    return dataset.map(_process_doc)
