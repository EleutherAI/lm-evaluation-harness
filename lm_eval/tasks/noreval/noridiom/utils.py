from collections import Counter
from string import punctuation

import numpy as np


def normalize(text):
    exclude = set(punctuation)
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


def process_results(doc, results):
    prediction = normalize(results[0])
    completions = [normalize(completion) for completion in doc["accepted_completions"]]
    exact_match = np.nanmax(
        [int(prediction == completion) for completion in completions]
    )
    fscore = np.nanmax(
        [f1(prediction=prediction, completion=completion) for completion in completions]
    )
    return {"em": exact_match, "fscore": fscore}


def filter_dataset_nb(dataset):
    return dataset.filter(lambda example: example["language"] == "nob")


def filter_dataset_nn(dataset):
    return dataset.filter(lambda example: example["language"] == "nno")
