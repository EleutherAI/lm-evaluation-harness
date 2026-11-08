"""Answer extraction for KorMedMCQA.

Implements the answer-extraction cascade from Appendix B of the KorMedMCQA
paper (https://arxiv.org/abs/2403.01469). Without it, ``exact_match`` is
computed on the raw generation, so any model that answers verbosely
(e.g. ``정답: B``) is scored 0 even when correct.
"""

import re

from lm_eval.api.filter import Filter


CHOICE_PATTERNS = [
    re.compile(r"정답[:\s]*([ABCDE])", re.IGNORECASE),
    re.compile(r"정답은\s*([ABCDE])\s*입니다", re.IGNORECASE),
    re.compile(r"\b([ABCDE])\.", re.IGNORECASE),
    re.compile(r"\b([ABCDE])\b", re.IGNORECASE),
]


def extract_choice(output: str) -> str:
    """Extract the chosen option letter (A-E) from a model generation.

    Tries each pattern of the paper's cascade in order and returns the last
    match of the first pattern that matches, or an empty string if no pattern
    matches.
    """
    for pattern in CHOICE_PATTERNS:
        matches = pattern.findall(output)
        if matches:
            return matches[-1].upper()
    return ""


class ExtractChoiceFilter(Filter):
    """Apply ``extract_choice`` to every model response."""

    def apply(self, resps, docs):
        return [[extract_choice(resp) for resp in r] for r in resps]
