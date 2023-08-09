import sklearn
import numpy as np


def atis_multi_f1(items):
    preds, golds = zip(*items)
    preds = np.array(preds)
    golds = np.array(golds)
    f11 = sklearn.metrics.f1_score(y_true=golds == 0, y_pred=preds == 0)
    f12 = sklearn.metrics.f1_score(y_true=golds == 1, y_pred=preds == 1)
    f13 = sklearn.metrics.f1_score(y_true=golds == 2, y_pred=preds == 2)
    f14 = sklearn.metrics.f1_score(y_true=golds == 3, y_pred=preds == 3)
    f15 = sklearn.metrics.f1_score(y_true=golds == 4, y_pred=preds == 4)
    f16 = sklearn.metrics.f1_score(y_true=golds == 5, y_pred=preds == 5)
    f17 = sklearn.metrics.f1_score(y_true=golds == 6, y_pred=preds == 6)
    f18 = sklearn.metrics.f1_score(y_true=golds == 7, y_pred=preds == 7)
    
    avg_f1 = np.mean([f11, f12, f13, f14, f15, f16, f17, f18])
    return avg_f1