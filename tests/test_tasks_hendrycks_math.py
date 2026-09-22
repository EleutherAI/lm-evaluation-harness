import pytest


utils = pytest.importorskip("lm_eval.tasks.hendrycks_math.utils")


GOLD_SOLUTION = (
    "Step 1: compute the total. Step 2: verify. "
    "Thus the answer is \\boxed{42}."
)


@pytest.mark.parametrize(
    "generation,expected",
    [
        # issue #2552: boxed answer inside \[...\] display math delimiters
        pytest.param(r"Thus the answer is \[ \boxed{42} \]", "42", id="display-math"),
        # boxed answer inside $...$ inline math
        pytest.param(r"Thus the answer is $ \boxed{42} $", "42", id="inline-math"),
        # boxed answer with no math delimiters at all
        pytest.param(r"Thus the answer is \boxed{42}.", "42", id="bare-boxed"),
        # the last boxed block wins, matching the gold-extraction direction
        pytest.param(r"x = \boxed{2} so y = \boxed{3}", "3", id="last-boxed-wins"),
        # \fbox is accepted as a LaTeX alternative
        pytest.param(r"Thus the answer is \fbox{7}", "7", id="fbox-accepted"),
    ],
)
def test_extract_flexible_answer_boxed_preferred(generation, expected):
    assert utils._extract_flexible_answer(generation) == expected


@pytest.mark.parametrize(
    "generation,expected",
    [
        # no boxed: fall back to the legacy $-slice
        pytest.param("The result is $ 7 $.", " 7 ", id="dollar-slice-fallback"),
        # no boxed, no dollars: pull the last number-like token
        pytest.param("The result is 42.", "42", id="bare-number"),
        pytest.param("It could be 2 or 3.5", "3.5", id="last-number"),
        pytest.param("answer: -7", "-7", id="negative-number"),
        # no boxed, no dollars, no number: the full generation as a last resort
        pytest.param("I cannot determine.", "I cannot determine.", id="full-string"),
    ],
)
def test_extract_flexible_answer_fallbacks(generation, expected):
    assert utils._extract_flexible_answer(generation) == expected


@pytest.mark.parametrize(
    "generation,expected",
    [
        pytest.param(r"\[ \boxed{42} \]", r"\[ \boxed{42} \]", id="display-math"),
        pytest.param("$ 42 $", " 42 ", id="dollar-slice"),
        pytest.param("The answer is 42", "The answer is 42", id="no-dollars"),
        pytest.param(r"\boxed{42}", r"\boxed{42}", id="bare-boxed"),
    ],
)
def test_extract_strict_answer_unchanged(generation, expected):
    """The strict extractor must preserve the historical exact_match behavior."""
    assert utils._extract_strict_answer(generation) == expected


@pytest.mark.parametrize(
    "generation,exact,flexible",
    [
        # the issue's example is missed by exact_match but caught leniently
        pytest.param(r"Thus the answer is \[ \boxed{42} \]", 0, 1, id="issue-2552"),
        pytest.param("The result is $ 42 $.", 1, 1, id="dollar-case"),
        pytest.param("42", 1, 1, id="bare-answer"),
        pytest.param("The total is 41.", 0, 0, id="wrong-answer"),
        pytest.param(r"Thus the answer is \boxed{43}.", 0, 0, id="wrong-boxed"),
    ],
)
def test_process_results_metrics(generation, exact, flexible):
    doc = {"solution": GOLD_SOLUTION}
    res = utils.process_results(doc, [generation])
    assert res == {"exact_match": exact, "flexible_match": flexible}