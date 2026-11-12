"""Harden HumanEval/MBPP-style tests against always-true ``__eq__`` returns."""

from __future__ import annotations

import re

# Injected into evaluated programs. Names are prefixed to avoid colliding with
# typical HumanEval / MBPP solutions.
PRELUDE = """
def __lm_eval_to_plain(value, _depth=0):
    if _depth > 32:
        raise ValueError("return value is too deeply nested")
    typ = type(value)
    if typ is type(None) or typ is bool or typ is str or typ is bytes:
        return value
    if typ is int:
        return int.__int__(value)
    if typ is float:
        return float.__float__(value)
    if typ is complex:
        return complex(value.real, value.imag)
    if typ is list:
        return [__lm_eval_to_plain(item, _depth + 1) for item in list.__iter__(value)]
    if typ is tuple:
        return tuple(__lm_eval_to_plain(item, _depth + 1) for item in tuple.__iter__(value))
    if typ is set:
        return {__lm_eval_to_plain(item, _depth + 1) for item in set.__iter__(value)}
    if typ is frozenset:
        return frozenset(__lm_eval_to_plain(item, _depth + 1) for item in frozenset.__iter__(value))
    if typ is dict:
        return {
            __lm_eval_to_plain(key, _depth + 1): __lm_eval_to_plain(val, _depth + 1)
            for key, val in dict.items(value)
        }
    raise TypeError("unsupported return type for evaluation")


def __lm_eval_wrap(fn):
    def __lm_eval_wrapped(*args, **kwargs):
        return __lm_eval_to_plain(fn(*args, **kwargs))

    return __lm_eval_wrapped
"""

# HumanEval references end with a top-level check(entry_point) call.
_CHECK_CALL = re.compile(r"(?m)^check\(([^()\n]+)\)\s*$")

# After MBPP (and similar) solutions run, wrap top-level user functions so
# `assert fn(...) == expected` cannot be satisfied by a custom __eq__.
WRAP_TOPLEVEL_FUNCTIONS = """
for __lm_eval_name, __lm_eval_obj in list(globals().items()):
    if __lm_eval_name.startswith("__lm_eval"):
        continue
    if type(__lm_eval_obj).__name__ == "function":
        globals()[__lm_eval_name] = __lm_eval_wrap(__lm_eval_obj)
"""


def harden_humaneval_reference(reference: str) -> str:
    """Wrap ``check(fn)`` so the candidate's return value is converted to builtins."""
    hardened, count = _CHECK_CALL.subn(r"check(__lm_eval_wrap(\1))", reference)
    if not count:
        return reference
    return PRELUDE + "\n" + hardened


def harden_humaneval_references(references: list[str]) -> list[str]:
    return [harden_humaneval_reference(reference) for reference in references]


def harden_mbpp_prediction(prediction: str) -> str:
    return prediction + "\n" + PRELUDE + "\n" + WRAP_TOPLEVEL_FUNCTIONS


def harden_mbpp_predictions(predictions: list[list[str]]) -> list[list[str]]:
    return [
        [harden_mbpp_prediction(prediction) for prediction in group]
        for group in predictions
    ]
