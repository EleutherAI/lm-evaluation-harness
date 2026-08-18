import re
import string
from collections import Counter


def _normalize_answer(text):
    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1_score(prediction, reference):
    prediction_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()

    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def process_results(doc, results):
    prediction = results[0].strip().split("\n", 1)[0]
    return {
        "f1": max(_f1_score(prediction, reference) for reference in doc["references"])
    }
