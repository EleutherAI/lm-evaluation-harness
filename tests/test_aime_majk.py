"""Tests for the AIME avg@k / maj@k metrics."""

import pytest

from lm_eval.tasks.aime.utils import process_results


DOC = {"problem": "?", "answer": "42", "id": "0"}


def answer(value: str) -> str:
    return f"so the answer is \\boxed{{{value}}}"


def test_unanimous_repeats():
    metrics = process_results(DOC, [answer("42")] * 5)
    assert metrics["exact_match"] == 1.0
    assert metrics["maj@k"] == 1.0


def test_majority_vote_wins_with_three_of_five():
    metrics = process_results(
        DOC, [answer("42"), answer("42"), answer("42"), answer("7"), answer("7")]
    )
    # avg@k sees 3/5 samples correct ...
    assert metrics["exact_match"] == pytest.approx(3 / 5)
    # ... and the vote is right
    assert metrics["maj@k"] == 1.0


def test_majority_vote_loses_with_two_of_five():
    metrics = process_results(
        DOC, [answer("42"), answer("42"), answer("7"), answer("7"), answer("7")]
    )
    assert metrics["exact_match"] == pytest.approx(2 / 5)
    assert metrics["maj@k"] == 0.0


def test_vote_is_taken_on_normalized_answers():
    # three spellings of the same answer are one vote, so they outvote the two 7s;
    # without normalization the vote would split 1-1-1 and the 7s would win
    metrics = process_results(
        DOC,
        [answer(" 42 "), answer("42"), answer("42  "), answer("7"), answer("7")],
    )
    assert metrics["exact_match"] == pytest.approx(3 / 5)
    assert metrics["maj@k"] == 1.0


def test_decimal_forms_are_not_merged_into_the_integer_answer():
    # documents the limit of the normalization: 42.0 does not count as 42, so it
    # neither scores nor votes with it
    metrics = process_results(
        DOC,
        [answer("42"), answer("42.0"), answer("7"), answer("7"), answer("7")],
    )
    assert metrics["exact_match"] == pytest.approx(1 / 5)
    assert metrics["maj@k"] == 0.0


def test_single_sample_matches_exact_match():
    assert process_results(DOC, [answer("42")])["maj@k"] == 1.0
    assert process_results(DOC, [answer("7")])["maj@k"] == 0.0


def test_answer_extracted_from_dollar_span_without_boxed():
    metrics = process_results(DOC, ["thus $42$", "thus $42$", "thus $7$"])
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert metrics["maj@k"] == 1.0


def test_unparseable_responses_never_vote_correct():
    metrics = process_results(DOC, ["no idea", "still none", "no"])
    assert metrics["exact_match"] == 0.0
    assert metrics["maj@k"] == 0.0


def test_nested_repeat_shape_is_flattened():
    # the harness can hand process_results one list per repeat
    metrics = process_results(DOC, [[answer("42")], [answer("42")], [answer("7")]])
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert metrics["maj@k"] == 1.0
