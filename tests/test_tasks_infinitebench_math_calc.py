import pytest


utils = pytest.importorskip("lm_eval.tasks.infinitebench.utils")

process = utils.process_results_math_calc


def doc(answer):
    return {"answer": answer}


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        # a perfect answer scores 1.0
        pytest.param([1, 3, 5, 2], "[1, 3, 5, 2]", 1.0, id="perfect-answer"),
        # brackets and commas are stripped, so plain spacing works too
        pytest.param([1, 3, 5, 2], "1 3 5 2", 1.0, id="whitespace-separated"),
        # answers may arrive as strings after HF dataset casting
        pytest.param(["1", "3", "5", "2"], "[1, 3, 5, 2]", 1.0, id="string-label"),
        # official scorer unwraps a single nested list
        pytest.param([[1, 3, 5, 2]], "[1, 3, 5, 2]", 1.0, id="nested-label"),
        # a stopped-short prediction still gets credit for its prefix
        pytest.param([1, 3, 5, 2], "[1, 3, 5]", 0.75, id="truncated-prefix"),
        # mismatch at the very first step scores zero
        pytest.param([1, 3, 5, 2], "[9, 3, 5, 2]", 0.0, id="first-step-mismatch"),
        # once the prefix breaks, later correct numbers are ignored
        pytest.param([1, 3, 5, 2], "[1, 9, 9, 9, 5, 2]", 0.25, id="broken-prefix"),
        # empty or non-numeric predictions score zero
        pytest.param([1, 3, 5, 2], "", 0.0, id="empty-prediction"),
        pytest.param([1, 3, 5, 2], "who knows", 0.0, id="garbage-prediction"),
        # official quirk: the digit split drops '-' so negative running
        # totals never match, even when the model is exactly right
        pytest.param([1, 3, -1, -11], "[1, 3, -1, -11]", 0.5, id="negative-sign-loss"),
    ],
)
def test_process_results_math_calc(label, prediction, expected):
    assert process(doc(label), [prediction]) == {"score": expected}


def test_missing_answer_scores_zero():
    assert process(doc([]), ["[1, 3, 5, 2]"]) == {"score": 0.0}