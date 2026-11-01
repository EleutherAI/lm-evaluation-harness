import pytest

from lm_eval.tasks.hrm8k.default import utils as korean
from lm_eval.tasks.hrm8k.en import utils as english


@pytest.fixture(params=[korean, english], ids=["korean", "english"])
def grader(request):
    return request.param


@pytest.mark.parametrize(
    "gold, prediction, expected",
    [
        (r"\frac{1}{6}", r"\frac{1}{6}", 1),
        (r"\frac{1}{6}", "6", 0),
        (r"8\pi", r"8\pi", 1),
        (r"8\pi", "8", 0),
        (r"\frac{x-3}{4}", r"\frac{x-3}{4}", 1),
        (r"\frac{x-3}{4}", "4", 0),
        ("-10.0", "-10", 1),
        ("-10.0", "10", 0),
        ("-3.0", "-3", 1),
        ("-3.0", "3", 0),
        (r"\frac{1}{2}", "0.5", 1),
        (r"\frac{1}{6}", "1/6", 1),
        ("18", "18.0", 1),
        ("18.0", "18", 1),
        ("18", "19", 0),
    ],
)
def test_reference_is_not_parsed_as_generated_text(grader, gold, prediction, expected):
    assert grader.process_results(
        {"answer": gold}, [r"\boxed{" + prediction + "}"]
    ) == {"exact_match": expected}


@pytest.mark.parametrize("response", ["I do not know", "", r"\boxed{unfinished"])
def test_unparseable_response_is_incorrect(grader, response):
    # A symbolic reference is valid even when it contains no bare number.
    assert grader.process_results({"answer": r"\pi"}, [response]) == {"exact_match": 0}


@pytest.mark.parametrize("response", [r"\boxed{18}", "$18$", "answer = 18", "18"])
def test_prediction_extraction_still_runs(grader, response):
    assert grader.process_results({"answer": "18"}, [response]) == {"exact_match": 1}
