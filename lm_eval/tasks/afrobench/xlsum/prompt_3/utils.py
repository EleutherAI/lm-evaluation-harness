import evaluate


def rougeL(items):
    """
    # passthrough for efficiency
    """
    return items


def rougeL_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rougeL"]
