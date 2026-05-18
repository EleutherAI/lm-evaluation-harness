from lm_eval.tasks.hendrycks_math.utils import (
    find_all_boxed_strings,
    process_results,
)


def _doc(boxed):
    return {"solution": f"... so \\boxed{{{boxed}}}."}


def test_boxed_response_matches():
    assert process_results(_doc("[2,5)"), ["The domain is \\boxed{[2,5)}"]) == {
        "exact_match": 1
    }


def test_dollar_fallback_still_works():
    assert process_results(
        _doc("\\frac{1}{2}"), ["The answer is $\\frac{1}{2}$"]
    ) == {"exact_match": 1}


def test_space_form_boxed_falls_back_to_last_boxed_only_string():
    # \boxed 4 (space-form) is intentionally skipped by find_all_boxed_strings;
    # last_boxed_only_string handles it, terminating on $ or end-of-string.
    assert process_results(_doc("4"), ["The answer is \\boxed 4"]) == {
        "exact_match": 1
    }


def test_space_form_boxed_terminated_by_dollar():
    # Same fallback path, but terminated by a $ delimiter (the way solutions
    # in the dataset typically wrap the answer).
    assert process_results(_doc("7"), ["So we get $\\boxed 7$ as the answer."]) == {
        "exact_match": 1
    }


def test_multi_boxed_joined():
    assert process_results(
        {"solution": "... \\boxed{3, 5, 7}."},
        ["Final answers: \\boxed{3}, \\boxed{5}, \\boxed{7}"],
    ) == {"exact_match": 1}


def test_multi_boxed_deduplicated():
    # Models often repeat the final answer; dedup keeps a single \boxed{4}
    # from producing "4, 4".
    assert process_results(_doc("4"), ["So \\boxed{4}. Therefore \\boxed{4}."]) == {
        "exact_match": 1
    }


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
    assert process_results(_doc("42"), ["I don't know"]) == {"exact_match": 0}
