import pytest

from lm_eval.tasks.hrm8k.default.utils import (
    postprocess as postprocess_default,
    process_results as process_results_default,
)
from lm_eval.tasks.hrm8k.en.utils import (
    postprocess as postprocess_en,
    process_results as process_results_en,
)


@pytest.mark.parametrize(
    "grader_module",
    [
        ("default", process_results_default),
        ("en", process_results_en),
    ],
)
@pytest.mark.parametrize(
    "gold, prediction, expected",
    [
        # Distinct large integers (> 2**53 and 10**16) that would collide under binary64 float
        ("9007199254740993", "9007199254740992", 0),
        ("9007199254740992", "9007199254740993", 0),
        ("10000000000000001", "10000000000000000", 0),
        ("10000000000000000", "10000000000000001", 0),
        # Identical large integers match
        ("9007199254740993", "9007199254740993", 1),
        ("10000000000000001", "10000000000000001", 1),
        # Standard integers
        ("18", "19", 0),
        ("18", "18", 1),
        # Float vs integer equivalence for whole numbers
        ("18.0", "18", 1),
        ("18", "18.0", 1),
        ("9007199254740993.0", "9007199254740993", 1),
        ("9007199254740993", "9007199254740993.0", 1),
    ],
)
def test_hrm8k_process_results_precision(grader_module, gold, prediction, expected):
    lang, process_results = grader_module
    response = f"\\boxed{{{prediction}}}"
    result = process_results({"answer": gold}, [response])
    assert result["exact_match"] == expected, (
        f"Failed for {lang}: gold={gold}, pred={prediction}"
    )


@pytest.mark.parametrize("postprocess", [postprocess_default, postprocess_en])
@pytest.mark.parametrize(
    "input_val, expected",
    [
        ("9007199254740993", "9007199254740993"),
        ("9007199254740992", "9007199254740992"),
        ("10000000000000001", "10000000000000001"),
        ("18.0", "18"),
        ("18.5", "18.5"),
        ("18.50", "18.5"),
        ("0.0", "0"),
        ("-0.0", "0"),
        ("\\frac{1}{2}", "\\frac{1}{2}"),
    ],
)
def test_hrm8k_postprocess(postprocess, input_val, expected):
    assert postprocess(input_val) == expected
