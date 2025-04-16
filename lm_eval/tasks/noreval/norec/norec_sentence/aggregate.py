import numpy as np
import sklearn


def multi_f1(items):
    """
    Computes the macro-average F1 score.
    """
    preds, golds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    fscore = sklearn.metrics.f1_score(golds, preds, average="macro")
    return fscore
