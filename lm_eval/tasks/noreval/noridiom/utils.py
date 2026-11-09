from collections import Counter
from string import punctuation

import numpy as np


def get_first_word(text):
    words = "".join(ch for ch in text if ch.isalpha() or ch.isspace()).lower().strip()
    if len(words) == 0:
        return ""
    return words.split()[0]


def normalize(text):
    exclude = set(punctuation)
    text = text.split("\n")[0]  # use only the first line for backwards compatibility
    return "".join(ch for ch in text if ch not in exclude).lower().strip()


def f1(prediction, completion):
    gold_toks = completion.split()
    pred_toks = prediction.split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _score_one(doc, completion_text):
    prediction = normalize(completion_text)
    prediction_first_word = get_first_word(completion_text)
    completions = [normalize(completion) for completion in doc["accepted_completions"]]

    exact_match_first_word = np.nanmax(
        [int(prediction_first_word == completion) for completion in completions]
    )
    exact_match = np.nanmax(
        [int(prediction == completion) for completion in completions]
    )
    fscore = np.nanmax(
        [f1(prediction=prediction, completion=completion) for completion in completions]
    )
    return {"em_first": exact_match_first_word, "em": exact_match, "fscore": fscore}


def process_results(doc, results):
    # results[0] holds the K sampled generations; average their scores
    scored = [_score_one(doc, completion) for completion in results[0]]
    return {key: sum(s[key] for s in scored) / len(scored) for key in scored[0]}
