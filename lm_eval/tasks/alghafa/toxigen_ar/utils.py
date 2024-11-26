import numpy as np


def doc_to_target(doc):
    return np.round(((doc["toxicity_ai"] + doc["toxicity_human"]) > 5.5)).astype(
        np.int32
    )
