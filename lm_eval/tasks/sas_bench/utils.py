import re
from sklearn.metrics import cohen_kappa_score


def extract_score(prediction, max_score):
    """Pull an integer score out of the model's generation."""
    text = str(prediction).strip()
    patterns = [
        r"[Ss]core[:\s]+(\d+)",            # "Score: 5"
        r"总分[：:]\s*(\d+)",               # Chinese "Total score: 5"
        r"(\d+)\s*(?:分|points|/|out of)",  # "5分", "5 points"
        r"(\d+)",                          # fallback: first integer anywhere
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return max(0, min(int(m.group(1)), int(max_score)))
    return 0  # no parseable number -> treated as 0 (documented limitation)


def doc_to_text(doc):
    steps = doc.get("steps") or []
    response = "\n".join(
        f"Step {i}: {s.get('response', '').strip()}"
        for i, s in enumerate(steps, 1)
        if s.get("response", "").strip()
    )
    return (
        "You are an expert grader. Read the question, the reference answer, "
        "and the student's response below. Output a single integer score "
        f"between 0 and {doc['total']}.\n\n"
        f"Question:\n{doc['question']}\n\n"
        f"Reference Answer:\n{doc['reference']}\n\n"
        f"Student Response:\n{response}\n\n"
        f"Max Score: {doc['total']}\n"
        "Score (integer only):"
    )


def process_results(doc, results):
    gold = int(doc["manual_label"])
    pred = extract_score(results[0], doc["total"])
    return {
        "exact_match": float(pred == gold),
        "qwk": (gold, pred),
    }


def qwk_agg(items):
    """items: list of (gold, pred) tuples for the whole dataset."""
    golds = [int(g) for g, _ in items]
    preds = [int(p) for _, p in items]
    if len(set(golds)) < 2 or len(set(preds)) < 2:
        # kappa is undefined/unstable with no variance -> fall back
        return sum(g == p for g, p in zip(golds, preds)) / len(golds)
    score = cohen_kappa_score(golds, preds, weights="quadratic")
    return float(score) if score == score else 0.0  # NaN guard