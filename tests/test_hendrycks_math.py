import pytest

from lm_eval.tasks.hendrycks_math.utils import (
    HAS_MATH_VERIFY,
    HAS_SYMPY,
    find_all_boxed_strings,
    is_equiv,
    process_results,
)


def _doc(boxed):
    return {"solution": f"... so \\boxed{{{boxed}}}."}


def _exact_match(doc, resps):
    # process_results also reports an optional `math_verify` key when
    # math_verify is installed, so assert on exact_match alone.
    return process_results(doc, resps)["exact_match"]


def test_boxed_response_matches():
    assert _exact_match(_doc("[2,5)"), ["The domain is \\boxed{[2,5)}"]) == 1


def test_dollar_fallback_still_works():
    assert _exact_match(_doc("\\frac{1}{2}"), ["The answer is $\\frac{1}{2}$"]) == 1


def test_space_form_boxed_falls_back_to_last_boxed_only_string():
    # \boxed 4 (space-form) is intentionally skipped by find_all_boxed_strings;
    # last_boxed_only_string handles it, terminating on $ or end-of-string.
    assert _exact_match(_doc("4"), ["The answer is \\boxed 4"]) == 1


def test_space_form_boxed_terminated_by_dollar():
    # Same fallback path, but terminated by a $ delimiter (the way solutions
    # in the dataset typically wrap the answer).
    assert _exact_match(_doc("7"), ["So we get $\\boxed 7$ as the answer."]) == 1


def test_multi_boxed_joined():
    assert (
        _exact_match(
            {"solution": "... \\boxed{3, 5, 7}."},
            ["Final answers: \\boxed{3}, \\boxed{5}, \\boxed{7}"],
        )
        == 1
    )


def test_multi_boxed_deduplicated():
    # Models often repeat the final answer; dedup keeps a single \boxed{4}
    # from producing "4, 4".
    assert _exact_match(_doc("4"), ["So \\boxed{4}. Therefore \\boxed{4}."]) == 1


def test_find_all_boxed_strings_returns_all_occurrences():
    # No dedup at this layer -- dedup is process_results' job.
    assert find_all_boxed_strings("\\boxed{3}, \\boxed{5}, \\boxed{3}") == [
        "\\boxed{3}",
        "\\boxed{5}",
        "\\boxed{3}",
    ]


def test_find_all_boxed_strings_ignores_space_form():
    # Documents the intentional scope of this helper; space-form is handled
    # by the last_boxed_only_string fallback in process_results.
    assert find_all_boxed_strings("The answer is \\boxed 4.") == []


def test_neither_format_does_not_crash():
    assert _exact_match(_doc("42"), ["I don't know"]) == 0


def test_is_equiv_still_rejects_unequal_answers():
    # Guards against the SymPy fallback over-matching: these are not equivalent
    # regardless of whether sympy is installed.
    assert is_equiv("3", "4") is False
    assert is_equiv("x + 1", "x + 2") is False


@pytest.mark.skipif(not HAS_SYMPY, reason="requires sympy (pip install lm-eval[math])")
def test_is_equiv_sympy_symbolic_fallback():
    # String comparison fails on reordered terms; SymPy simplification catches it.
    assert is_equiv("9a + 11", "11 + 9a") is True


@pytest.mark.skipif(not HAS_SYMPY, reason="requires sympy (pip install lm-eval[math])")
def test_comma_separated_answers_are_not_symbolically_equated():
    # parse_latex truncates at the first comma ("3, 5, 7" -> 3), so without a
    # guard any two lists sharing a first element compare equal. MATH uses this
    # answer format ("separated by commas") a lot, and process_results joins
    # multiple \boxed{} values with ", ", so this must not over-match.
    assert is_equiv("1,-2", "1,-3") is False
    assert is_equiv("1,2", "1,3") is False
    assert is_equiv("3, 5, 7", "3, 5, 9") is False


@pytest.mark.skipif(not HAS_SYMPY, reason="requires sympy (pip install lm-eval[math])")
def test_thousands_separator_still_compares_numerically():
    # The comma guard must not disable digit-grouping commas, which parse_latex
    # handles correctly.
    assert is_equiv("58,500", "58500") is True


@pytest.mark.skipif(
    not HAS_MATH_VERIFY, reason="requires math_verify (pip install lm-eval[math])"
)
def test_math_verify_metric_reported():
    result = process_results(_doc("4"), ["The answer is \\boxed{4}"])
    assert result["math_verify"] == 1


@pytest.mark.skipif(HAS_MATH_VERIFY, reason="asserts behavior without math_verify")
def test_math_verify_metric_absent_without_dependency():
    result = process_results(_doc("4"), ["The answer is \\boxed{4}"])
    assert "math_verify" not in result
