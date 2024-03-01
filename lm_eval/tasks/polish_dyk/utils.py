import numpy as np
import sklearn.metrics


def f1(predictions, references):
    _prediction = predictions[0]
    _reference = references[0]
    string_label = ["B", "C"]
    reference = string_label.index(_reference)
    prediction = (
        string_label.index(_prediction)
        if _prediction in string_label
        else 0
    )

    return (prediction, reference)


def agg_f1(items):
    predictions, references = zip(*items)
    references, predictions = np.asarray(references), np.asarray(predictions)

    return sklearn.metrics.f1_score(references, predictions)