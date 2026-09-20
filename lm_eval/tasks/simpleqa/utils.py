"""
SimpleQA scoring utilities for lm-evaluation-harness.
"""

import re
import string

_NOT_ATTEMPTED_RE = re.compile(
    r"\b("
    r"i don'?t know|i do not know"
    r"|i'?m not sure|i am not sure"
    r"|i'?m not certain|i am not certain"
    r"|i cannot|i can'?t"
    r"|i'?m unable|i am unable"
    r"|i'?m not aware|i am not aware"
    r"|i have no (?:idea|information|knowledge)"
    r"|i don'?t have|i do not have"
    r"|cannot be determined"
    r"|i cannot answer|i can'?t answer"
    r"|i need more (?:information|context|data)"
    r"|i'?d need (?:more|additional)"
    r"|not (?:enough|sufficient) (?:information|context|data)"
    r")\b",
    re.IGNORECASE,
)

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE | re.IGNORECASE)


def normalize_answer(text):
    text = text.lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def classify_answer(prediction, gold):
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if not gold_norm:
        return "not_attempted"

    gold_tokens = set(gold_norm.split())
    pred_tokens = set(pred_norm.split())

    def _gold_is_present():
        if gold_norm in pred_norm:
            return True
        if gold_tokens == pred_tokens:
            return True
        if pred_norm in gold_norm and len(pred_tokens) >= max(1, len(gold_tokens) - 1):
            return True
        if gold_tokens and gold_tokens.issubset(pred_tokens):
            return True
        return False

    if _NOT_ATTEMPTED_RE.search(prediction):
        return "correct" if _gold_is_present() else "not_attempted"

    if gold_norm in pred_norm:
        return "correct"

    if gold_tokens == pred_tokens:
        return "correct"

    if pred_norm in gold_norm and len(pred_tokens) >= max(1, len(gold_tokens) - 1):
        return "correct"

    return "incorrect"


def process_results(doc, results):
    prediction = results[0]
    gold = doc["answer"]
    grade = classify_answer(prediction, gold)

    is_correct = int(grade == "correct")
    is_incorrect = int(grade == "incorrect")
    is_not_attempted = int(grade == "not_attempted")

    return {
        "correct": is_correct,
        "not_attempted": is_not_attempted,
        "f1": (is_correct, is_incorrect, is_not_attempted),
    }


def simpleqa_f1_agg(items):
    correct = sum(t[0] for t in items)
    incorrect = sum(t[1] for t in items)
    not_attempted = sum(t[2] for t in items)
    total = correct + incorrect + not_attempted

    if total == 0:
        return 0.0

    overall_correct = correct / total
    attempted = correct + incorrect
    acc_given_attempted = correct / attempted if attempted > 0 else 0.0

    denom = acc_given_attempted + overall_correct
    if denom == 0.0:
        return 0.0
    return 2.0 * acc_given_attempted * overall_correct / denom
