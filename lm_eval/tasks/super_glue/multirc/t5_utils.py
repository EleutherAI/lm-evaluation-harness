import collections

import numpy as np
import sklearn.metrics


def f1(predictions, references):  # This is a passthrough function
    _prediction = predictions[0]
    _reference = references[0].split("_")[-1]
    string_label = ["False", "True"]
    reference = string_label.index(_reference)
    prediction = (
        string_label.index(_prediction)
        if _prediction in string_label
        else not bool(reference)
    )

    return (prediction, reference)


def agg_f1(items):
    predictions, references = zip(*items)
    references, predictions = np.asarray(references), np.asarray(predictions)

    return sklearn.metrics.f1_score(references, predictions)


def em(predictions, references):  # This is a passthrough function
    _prediction = predictions[0]
    _group, _reference = references[0].split("_")
    string_label = ["False", "True"]
    reference = string_label.index(_reference)
    prediction = (
        string_label.index(_prediction)
        if _prediction in string_label
        else not bool(reference)
    )

    return (_group, prediction, reference)


def agg_em(items):
    grouped_values = collections.defaultdict(lambda: ([], []))
    for group, prediction, reference in items:
        grouped_values[group][0].append(reference)
        grouped_values[group][1].append(prediction)

    group_scores = []
    for group, (targets, predictions) in grouped_values.items():
        score = float(np.array_equal(targets, predictions))
        group_scores.append(score)

    return np.mean(group_scores)
