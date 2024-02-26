import numpy as np
import sklearn


def cb_multi_fi(items):
    preds, golds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
    f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
    f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
    avg_f1 = np.mean([f11, f12, f13])
    return avg_f1
