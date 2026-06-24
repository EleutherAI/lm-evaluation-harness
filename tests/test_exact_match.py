"""Tests for exact_match whitespace normalization.

Demonstrates that ignore_punctuation / ignore_numbers / regexes_to_ignore can
leave internal/edge whitespace behind, causing semantically identical answers
to be scored as mismatches, and that the opt-in ``ignore_whitespace`` flag
resolves this without changing default behavior.
"""
from lm_eval.api.metrics import exact_match_hf_evaluate


def _score(pred, ref, **kw):
    return exact_match_hf_evaluate([pred], [ref], **kw)["exact_match"]


# ---------------------------------------------------------------------------
# Regression: whitespace gap (default behavior, no ignore_whitespace)
# ---------------------------------------------------------------------------

def test_punctuation_strip_leaves_inner_whitespace_causing_mismatch():
    """REGRESSION: '( B )' vs '(B)' with ignore_punctuation should be the same
    answer, but leftover inner whitespace makes them mismatch."""
    assert _score("( B )", "(B)", ignore_punctuation=True) == 0.0


def test_punctuation_strip_leaves_edge_whitespace_causing_mismatch():
    """REGRESSION: leading/trailing space survives ignore_punctuation."""
    assert _score(" (B)", "(B)", ignore_punctuation=True) == 0.0
    assert _score("(B) ", "(B)", ignore_punctuation=True) == 0.0


def test_numbers_strip_leaves_inner_whitespace_causing_mismatch():
    """REGRESSION: '4 2' vs '42' with ignore_numbers mismatches."""
    assert _score("4 2", "42", ignore_numbers=True) == 0.0


def test_baseline_identical_strings_still_match():
    """SANITY: identical inputs still match with normalization flags on."""
    assert _score("(B)", "(B)", ignore_punctuation=True) == 1.0
    assert _score("42", "42", ignore_numbers=True) == 1.0


# ---------------------------------------------------------------------------
# Fix: ignore_whitespace resolves the gap
# ---------------------------------------------------------------------------

def test_ignore_whitespace_resolves_inner_whitespace_after_punct_strip():
    """FIX: ignore_whitespace collapses '( B )' and '(B)' to a match."""
    assert _score("( B )", "(B)", ignore_punctuation=True, ignore_whitespace=True) == 1.0


def test_ignore_whitespace_resolves_edge_whitespace():
    """FIX: leading/trailing/inner whitespace no longer causes mismatches."""
    assert _score(" (B) ", "(B)", ignore_punctuation=True, ignore_whitespace=True) == 1.0
    assert _score("( B ) ", "(B)", ignore_punctuation=True, ignore_whitespace=True) == 1.0


def test_ignore_whitespace_collapses_multiple_internal_spaces():
    """FIX: runs of spaces collapse to one, so '(B)    (C)' style outputs
    compare consistently with references."""
    assert (
        _score("a    b", "a b", ignore_whitespace=True) == 1.0
    )


def test_ignore_whitespace_preserves_distinct_answers():
    """FIX: distinct answers still don't match after whitespace normalization."""
    assert _score("(A)", "(B)", ignore_punctuation=True, ignore_whitespace=True) == 0.0


def test_ignore_whitespace_is_opt_in_and_does_not_change_defaults():
    """BACKWARD COMPAT: without ignore_whitespace, the legacy mismatch behavior
    is unchanged so existing task scores are not silently altered."""
    assert _score("( B )", "(B)", ignore_punctuation=True) == 0.0
    assert _score("( B )", "(B)", ignore_punctuation=True, ignore_whitespace=False) == 0.0
