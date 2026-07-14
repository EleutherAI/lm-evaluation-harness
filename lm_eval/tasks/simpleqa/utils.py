"""Answer processing for the SimpleQA task.

The original SimpleQA grades answers with an LLM judge (GPT-4o) into
correct / incorrect / not_attempted. This implementation is a deterministic,
offline approximation: SQuAD-style answer normalization for exact_match and
token-F1, plus a regex heuristic for not_attempted. It requires no API calls,
but differs from the reference grader in two ways worth noting:

- Exact match can under-credit verbose answers (e.g. "The answer is 1867"
  normalizes to "answer is 1867", not "1867"); f1 captures the partial overlap.
- not_attempted detection is English-only and pattern-based, so it may miss
  refusals phrased differently or in other languages.

Users who want the original model-graded setup can follow the judge pattern
used in the `gpqa` task.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(s: str) -> str:
    """Lower-case, drop punctuation and leading articles, collapse whitespace.

    Mirrors the SQuAD/TriviaQA-style normalization used for open-ended QA.
    """
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


# English refusal phrases. Heuristic approximation of the reference judge's
# "not attempted" category; see the module docstring for limitations.
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
    """Return True if the model explicitly declined to answer."""
    return bool(_NOT_ATTEMPTED_PATTERNS.search(model_output))


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between normalized prediction and reference (SQuAD-style)."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)  # both empty -> 1.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """Score one SimpleQA example into exact_match / f1 / not_attempted."""
    assert len(results) == 1, "SimpleQA is a single-answer task"
    model_output = results[0].strip()
    reference = doc["answer"]

    if is_not_attempted(model_output):
        return {"exact_match": 0.0, "f1": 0.0, "not_attempted": 1.0}

    exact = float(normalize_answer(model_output) == normalize_answer(reference))
    f1 = token_f1(model_output, reference)
    return {"exact_match": exact, "f1": f1, "not_attempted": 0.0}
