import pytest

from lm_eval.tasks.hrm8k.default import utils as korean
from lm_eval.tasks.hrm8k.en import utils as english


@pytest.fixture(params=[korean, english], ids=["korean", "english"])
def grader(request):
    return request.param


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("9007199254740993", "9007199254740993"),
        (9007199254740993, "9007199254740993"),
        # OMNI_MATH value reported in the review of PR #4494.
        (66071772829247409, "66071772829247409"),
        ("9007199254740993.0", "9007199254740993"),
        ("-9007199254740993", "-9007199254740993"),
        ("1.0000000000000001", "1.0000000000000001"),
        (
            "123456789012345678901234567890.123456789",
            "123456789012345678901234567890.123456789",
        ),
        (" 18.00 ", "18"),
        ("1.2500", "1.25"),
        ("1.8e1", "18"),
        ("-0.0", "0"),
        (r"\frac{1}{2}", r"\frac{1}{2}"),
    ],
)
def test_target_normalization_preserves_exact_value(grader, answer, expected):
    assert grader.doc_to_target({"answer": answer}) == expected


@pytest.mark.parametrize(
    "gold, prediction, expected",
    [
        ("9007199254740993", "9007199254740992", 0),
        ("9007199254740992", "9007199254740993", 0),
        ("10000000000000001", "10000000000000000", 0),
        ("66071772829247409", "66071772829247408", 0),
        ("66071772829247409", "66071772829247409", 1),
        ("9007199254740993", "9007199254740993", 1),
        ("9007199254740993.0", "9007199254740993", 1),
        ("1.0000000000000001", "1", 0),
        ("1", "1.0000000000000001", 0),
        ("1.0000000000000001", "1.0000000000000001", 1),
        ("18", "18.0", 1),
        ("18.0", "18", 1),
        ("18", "19", 0),
        ("1.25", "1.2500", 1),
        ("0.5", r"\frac{1}{2}", 1),
    ],
)
def test_boxed_answer_scoring(grader, gold, prediction, expected):
    response = r"\boxed{" + prediction + "}"
    assert grader.process_results({"answer": gold}, [response]) == {
        "exact_match": expected
    }
