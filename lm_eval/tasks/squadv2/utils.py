import evaluate

from math import exp
from functools import partial


def process_results(doc, results):

    continuation  = results[0]
    no_answer_probability = 0 # exp(logprob_unanswerable)

    predictions = {
        "id": doc["id"],
        "prediction_text": continuation,
        "no_answer_probability": no_answer_probability,
    }

    references = {
        "id": doc["id"],
        "answers": doc["answers"],
    }

    return {
        "predictions": predictions,
        "reference": references
    }
    # return _squad_metric([predictions], [references])
    # return {key: value if key in metrics for key, value in score.items()}


def _squad_metric(predictions, references):
    squad_metric = evaluate.load("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)

# Exact match (the normalized answer exactly match the gold answer)
def exact(items):
    print(items)
    import sys; sys.exit()
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references)["exact"]

# The F-score of predicted tokens versus the gold answer
def f1(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["f1"]

# Exact match (the normalized answer exactly match the gold answer)
def HasAns_exact(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["HasAns_exact"]

# The F-score of predicted tokens versus the gold answer
def HasAns_f1(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["HasAns_f1"]

# Exact match (the normalized answer exactly match the gold answer)
def NoAns_exact(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["NoAns_exact"]

# The F-score of predicted tokens versus the gold answer
def NoAns_f1(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["NoAns_f1"]

# Best exact match (with varying threshold)
def best_exact(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["best_exact"]

# Best F1 (with varying threshold)
def best_f1(predictions, references):
    return _squad_metric(predictions=predictions, references=references)["best_f1"]
