from sklearn.metrics import f1_score


def doc_to_text(doc):
    output = """Please identify whether the premise entails or contradicts the hypothesis in the following premise
    and hypothesis. The answer should be exact entailment, contradiction, or neutral.

    Premise: {premise}
    Hypothesis: {hypothesis}

    Is it entailment, contradiction, or neutral?"""

    text = output.format(premise=doc["premise"], hypothesis=doc["hypothesis"])
    return text


def doc_to_target(doc):
    replacements = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return replacements[doc["label"]]


def weighted_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore
