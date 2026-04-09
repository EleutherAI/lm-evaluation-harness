"""
utils.py — SimpleQA answer processing for lm-evaluation-harness

Concept: process_results()
──────────────────────────
This is the scoring function the harness calls for every example.
It receives:
  - doc:     the raw dataset row (has 'problem', 'answer', 'metadata')
  - results: list with one item — the model's generated string

It returns a dict mapping metric names → scores (0.0 or 1.0).
Those scores are averaged across all examples to produce the
final benchmark numbers.

Metrics we compute:
  1. exact_match      — 1.0 if normalized model output == normalized reference
  2. f1               — token-overlap F1 (partial credit, like SQuAD)
  3. not_attempted    — 1.0 if the model explicitly refused to answer
"""

import re
import string
from collections import Counter
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Answer Normalization
# ─────────────────────────────────────────────────────────────────────────────
# Raw model output might be "  The year 1867. " while the reference is "1867".
# Normalization strips noise so we're comparing the semantic content.
# This is standard practice for open-ended QA (SQuAD, TriviaQA, NQ all do it).

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

def normalize_answer(s: str) -> str:
    """
    Lower-case, remove punctuation, strip leading articles, collapse whitespace.
    Mirrors the normalization used in the original SimpleQA evaluation code.
    """
    # 1. Lowercase everything
    s = s.lower()
    # 2. Remove punctuation  e.g.  "1867."  →  "1867"
    s = s.translate(str.maketrans("", "", string.punctuation))
    # 3. Drop leading articles  e.g.  "the eiffel tower"  →  "eiffel tower"
    s = _ARTICLES.sub(" ", s)
    # 4. Collapse multiple spaces into one
    s = " ".join(s.split())
    return s


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Not-Attempted Detection
# ─────────────────────────────────────────────────────────────────────────────
# SimpleQA measures three states: correct, incorrect, not_attempted.
# "Not attempted" = the model says it doesn't know.
# This is IMPORTANT: a model that hedges is safer than one that confidently
# hallucinates. Tracking this rate separately gives a richer picture of
# model behavior than just accuracy alone.

_NOT_ATTEMPTED_PATTERNS = re.compile(
    r"(i don'?t know"
    r"|i'?m not sure"
    r"|i cannot (determine|say|provide|tell)"
    r"|i do not (know|have)"
    r"|cannot be determined"
    r"|not (enough|sufficient) information"
    r"|unclear from"
    r"|i'?m unable to"
    r"|no information"
    r"|this (question|answer) (is beyond|cannot))",
    re.IGNORECASE,
)

def is_not_attempted(model_output: str) -> bool:
    """Returns True if the model explicitly declined to answer."""
    return bool(_NOT_ATTEMPTED_PATTERNS.search(model_output))


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: Token-level F1
# ─────────────────────────────────────────────────────────────────────────────
# Exact match is strict: "Marie Curie" ≠ "Curie".
# F1 gives partial credit by counting token overlap (like SQuAD evaluation).
# Both F1 = 0.0 and exact_match = 0.0 means total failure.
# F1 > 0 but exact_match = 0 means the model got part of it right.

def token_f1(prediction: str, reference: str) -> float:
    """
    Token-overlap F1 between normalized prediction and reference.
    Standard SQuAD-style metric.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)  # both empty → 1.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT: process_results — the harness entry point
# ─────────────────────────────────────────────────────────────────────────────
# The harness calls this once per example after the model generates an answer.
# Return value: dict[metric_name → float]
# The harness then averages each metric across all examples.

def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """
    Score one SimpleQA example.

    Args:
        doc:     Dataset row with keys 'problem', 'answer', 'metadata'
        results: [model_output]  (list with exactly one string)

    Returns:
        {
            "exact_match":   1.0 or 0.0,
            "f1":            0.0 – 1.0,
            "not_attempted": 1.0 or 0.0,
        }
    """
    assert len(results) == 1, "SimpleQA is a single-answer task"
    model_output: str = results[0].strip()
    reference: str = doc["answer"]

    # ── Step 1: Did the model refuse to answer? ───────────────
    # If so, mark as not_attempted and short-circuit.
    if is_not_attempted(model_output):
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "not_attempted": 1.0,
        }

    # ── Step 2: Normalize both strings ────────────────────────
    norm_pred = normalize_answer(model_output)
    norm_ref = normalize_answer(reference)

    # ── Step 3: Exact match ───────────────────────────────────
    # After normalization, are the strings identical?
    exact = float(norm_pred == norm_ref)

    # ── Step 4: Token F1 ─────────────────────────────────────
    f1 = token_f1(model_output, reference)

    return {
        "exact_match": exact,
        "f1": f1,
        "not_attempted": 0.0,
    }
