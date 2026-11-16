from lm_eval.utils import weighted_f1_score


def doc_to_target(doc):
    replacements = {0: "True", 1: "Neither", 2: "False"}
    return replacements[doc["label"]]
