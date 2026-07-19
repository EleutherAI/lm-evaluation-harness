import datasets
import numpy as np
import sklearn


label_map = {0: "negative", 1: "neutral", 2: "positive"}


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["choices"] = ["negative", "neutral", "positive"]

        doc["label"] = label_map[doc["label"]]

        return doc

    return dataset.map(_helper)


def f1(predictions, references):
    return (predictions[0], references[0])


def agg_f1_weighted(items):
    predictions, references = zip(*items)
    references, predictions = np.asarray(references), np.asarray(predictions)

    fscore = sklearn.metrics.f1_score(references, predictions, average="weighted")
    return fscore
