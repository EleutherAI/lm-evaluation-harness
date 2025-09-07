# preprocess_winogrande_var.py
# Path: lm_eval/tasks/winogrande_var/preprocess_winogrande_var.py

import re

# ------------- helpers ------------- #

_PUNCT_EDGES = r'^[\'"“”‘’(]+|[\'"“”‘’),.:;!?]+$'

def _normalize(text: str) -> str:
    if text is None:
        return ""
    s = text.strip().lower()
    s = re.sub(_PUNCT_EDGES, "", s)
    return s

def _first_token(text: str) -> str:
    s = _normalize(text)
    return s.split()[0] if s else s

# ------------- task API functions ------------- #

def doc_to_text(doc):
    """
    Build a cloze-style prompt with the two options in parentheses (soft word-bank).
    Example output:

    "The trophy doesn’t fit in the suitcase because _____ is too small.
     (choices: the trophy / the suitcase)
     Answer (fill the blank with the correct word/name only): "
    """
    sent = doc["sentence"]
    o1, o2 = doc["option1"], doc["option2"]

    # Turn the underscore into a visible blank for clarity
    prompt_sentence = sent.replace("_", "_____")

    prompt = (
        f"{prompt_sentence}\n"
        f"(choices: {o1} / {o2})\n"
        "Answer (fill the blank with the correct word/name only): "
    )
    return prompt


def doc_to_target(doc):
    """
    Return the gold surface form string, not the numeric label.
    """
    correct = doc["option1"] if str(doc["answer"]) == "1" else doc["option2"]
    return correct


def process_results(doc, results):
    """
    results: the raw generation string returned by the model for this doc.

    We compute:
      - exact_match: normalized generation equals normalized target
      - prefix_match@1: first token of generation equals first token of target
        (gives partial credit if the model prints e.g. 'amy' vs 'amy.' or 'amy is')
    """
    target = doc_to_target(doc)
    pred_raw = results

    # normalize both
    gold_norm = _normalize(target)
    pred_norm = _normalize(pred_raw)

    # exact-match on the full normalized string
    em = 1.0 if pred_norm == gold_norm else 0.0

    # prefix@1: first token match
    gold_first = _first_token(target)
    pred_first = _first_token(pred_raw)
    prefix = 1.0 if (pred_first and gold_first and pred_first == gold_first) else 0.0

    return {
        "exact_match": em,
        "prefix_match@1": prefix,
    }
