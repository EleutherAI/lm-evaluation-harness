import re

from scipy.stats import pearsonr


def parse_score(text: str):
    """Extract a float in [1, 5] from model output; return None on failure."""
    match = re.search(r"\b([1-5](?:[.,]\d+)?)\b", text)
    if match:
        score = float(match.group(1).replace(",", "."))
        return max(1.0, min(5.0, score))
    return None


def process_results_sts(doc, results):
    pred = parse_score(results[0].strip())
    if pred is None:
        pred = 3.0  # neutral fallback when model output is unparseable
    gold = float(doc["relatedness_score"])
    return {"pearsonr": (pred, gold)}


def pearson_corr(items):
    preds = [i[0] for i in items]
    golds = [i[1] for i in items]
    if len(set(preds)) < 2:
        return 0.0
    return float(pearsonr(preds, golds)[0])
