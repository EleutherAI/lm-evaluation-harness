import datasets
import pytest

from lm_eval.tasks.theoremqa.utils import (
    extract_answer,
    is_correct,
    list_fewshot_samples,
    process_docs,
    process_results,
)


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("The answer is 833.33", "833.33"),
        ("...so it follows.\nThe answer is True.", "True"),
        # the last trigger wins, as in the reference implementation
        ("The answer is 3\nThe answer is 4", "4"),
        # a continued few-shot block stops at the next question
        ("The answer is 7\n\nQuestion: something else", "7"),
        # no trigger: fall back to the final line
        ("Working through it\n11760", "11760"),
        ("", ""),
    ],
)
def test_extract_answer(completion, expected):
    assert extract_answer(completion) == expected


@pytest.mark.parametrize(
    "prediction, gold, answer_type, expected",
    [
        # floats carry the reference's 4% relative window
        ("833.33", "833.33", "float", True),
        ("850", "833.33", "float", True),
        ("900", "833.33", "float", False),
        ("-0.98", "-1.0", "float", True),
        # integers compare after rounding, so the window does not apply
        ("11760", "11760", "integer", True),
        ("11759.6", "11760", "integer", True),
        ("11800", "11760", "integer", False),
        # fractions and units are parsed rather than rejected
        ("10/3", "3.33", "float", True),
        ("$42", "42", "integer", True),
        ("5%", "5", "integer", True),
        # bool accepts yes/no phrasing, and refuses an answer claiming both
        ("True", "True", "bool", True),
        ("Yes, such a graph exists", "True", "bool", True),
        ("No", "True", "bool", False),
        ("False", "False", "bool", True),
        ("true or false depending on the case", "True", "bool", False),
        # options match on the labelled choice anywhere in the span
        ("(a)", "(a)", "option", True),
        ("the correct option is (A)", "(a)", "option", True),
        ("(b)", "(a)", "option", False),
        # an answer naming every option must not pass on containment alone
        ("(a), (b), (c), (d)", "(a)", "option", False),
        # lists compare elementwise after sorting, and length must match
        ("[1, 2, 3]", "[3, 2, 1]", "list of integer", True),
        ("[1, 2]", "[1, 2, 3]", "list of integer", False),
        ("[3.33, 1.33]", "[1.33, 3.33]", "list of float", True),
        ("(3.33, 1.33)", "[1.33, 3.33]", "list of float", True),
        ("3.33, 1.33", "[1.33, 3.33]", "list of float", False),
        # unparsable predictions score wrong instead of raising
        ("\\frac{10}{3}", "3.33", "float", False),
        ("", "3.33", "float", False),
    ],
)
def test_is_correct(prediction, gold, answer_type, expected):
    assert is_correct(prediction, gold, answer_type) is expected


def test_process_results_scores_one_doc():
    doc = {"Question": "q", "Answer": "833.33", "Answer_type": "float"}
    assert process_results(doc, ["The answer is 850"]) == {"exact_match": 1.0}
    assert process_results(doc, ["The answer is 900"]) == {"exact_match": 0.0}


def test_process_docs_drops_image_rows_and_column():
    dataset = datasets.Dataset.from_dict(
        {
            "Question": ["text only", "needs a figure"],
            "Answer": ["1", "2"],
            "Answer_type": ["integer", "integer"],
            "Picture": [None, {"bytes": b"not-a-real-image", "path": None}],
        },
        features=datasets.Features(
            {
                "Question": datasets.Value("string"),
                "Answer": datasets.Value("string"),
                "Answer_type": datasets.Value("string"),
                "Picture": datasets.Image(decode=False),
            }
        ),
    )

    processed = process_docs(dataset)

    assert "Picture" not in processed.column_names
    assert processed["Question"] == ["text only"]


def test_fewshot_samples_carry_a_target_and_a_flag():
    samples = list_fewshot_samples()
    assert len(samples) == 5
    for sample in samples:
        assert sample["few_shot"] == "1"
        assert sample["Solution"].strip().splitlines()[-1].startswith("The answer is")
