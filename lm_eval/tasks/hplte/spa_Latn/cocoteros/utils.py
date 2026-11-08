import evaluate


def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    transposed = list(zip(*items, strict=False))
    refs, preds = transposed[0], transposed[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]
