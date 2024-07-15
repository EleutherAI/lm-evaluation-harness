from sklearn.metrics import f1_score


def doc_to_target(doc):
    replacements = {0: "True", 1: "Neither", 2: "False"}
    return replacements[doc["label"]]


def weighted_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore
