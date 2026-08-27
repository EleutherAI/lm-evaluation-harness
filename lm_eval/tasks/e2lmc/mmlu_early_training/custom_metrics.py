import numpy as np


def loglikelihood_diff(items):
    diffs = []
    for item in items:
        target, lls = item
        target_ll = lls[target]
        others_ll = [ll for i, ll in enumerate(lls) if i != target]
        mean_others_ll = np.mean(others_ll)
        diff = target_ll - mean_others_ll
        diffs.append(diff)

    return np.mean(diffs)
