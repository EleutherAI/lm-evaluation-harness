import re
from sklearn.metrics import cohen_kappa_score


def extract_score(prediction: str, max_score: int) -> int:
    """Extract numeric score from model output."""
    patterns = [
        r"^\s*(\d+)",                          # number at very start
        r"[Ss]core[:\s]+(\d+)",               # "Score: 5"
        r"总分[：:]\s*(\d+)",                   # Chinese "Total score: 5"
        r"(\d+)\s*(?:分|points|/|out of)",     # "5分" or "5 points"
    ]
    for pattern in patterns:
        match = re.search(pattern, prediction.strip())
        if match:
            val = int(match.group(1))
            return min(val, max_score)
    return 0


def process_steps(steps):
    """Join all student step responses into one string."""
    if not steps:
        return ""
    parts = []
    for i, step in enumerate(steps, 1):
        response = step.get("response", "").strip()
        if response:
            parts.append(f"Step {i}: {response}")
    return "\n".join(parts)


def qwk_agg(items):
    """Aggregation function — receives list of (reference, prediction) tuples."""
    refs = [int(item[0]) for item in items]
    preds = [item[1] for item in items]

    if len(set(refs)) < 2:
        # QWK undefined with one unique class — fall back to exact match
        return sum(r == p for r, p in zip(refs, preds)) / len(refs)

    return float(cohen_kappa_score(refs, preds, weights="quadratic"))


def qwk_score(references, predictions, **kwargs):
    """Called by lm-eval with lists of references and predictions."""
    max_scores = kwargs.get("max_scores", [100] * len(predictions))
    refs = [int(r) for r in references]
    preds = [
        extract_score(str(p), max_scores[i] if i < len(max_scores) else 100)
        for i, p in enumerate(predictions)
    ]

    if len(set(refs)) < 2:
        return sum(r == p for r, p in zip(refs, preds)) / len(refs)

    return float(cohen_kappa_score(refs, preds, weights="quadratic"))