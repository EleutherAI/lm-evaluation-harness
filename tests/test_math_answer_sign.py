"""A number read out of a free-text answer keeps its sign.

`parse_math_answer` reads an answer through one of several paths and falls back to
a regex over the raw text when the answer is neither boxed, nor wrapped in `$...$`,
nor preceded by an `=`. That regex did not accept a leading minus, so a negative
gold was reduced to its absolute value and a model answering with the wrong sign was
scored correct:

    is_equiv("2", "-2")  ->  True   (before)
    is_equiv("2", "-2")  ->  False  (after)

AGIEval MATH and HRM8K each carry their own copy of the helper, so every copy is
covered. The cases are grouped by the dimension they vary:

* the path the answer is read through (bare, boxed, dollar, `=`),
* the shape of the value (single and multi digit, decimal, negative),
* the same value reached through two different paths,
* the gold answers the two datasets actually contain,
* the verdicts that have to stay exactly as they were.
"""

import pytest

from lm_eval.tasks.agieval.utils import is_equiv as agieval_is_equiv
from lm_eval.tasks.hrm8k.default.utils import is_equiv as hrm8k_default_is_equiv
from lm_eval.tasks.hrm8k.en.utils import is_equiv as hrm8k_en_is_equiv


IS_EQUIV = {
    "agieval": agieval_is_equiv,
    "hrm8k/default": hrm8k_default_is_equiv,
    "hrm8k/en": hrm8k_en_is_equiv,
}


@pytest.mark.parametrize("name", IS_EQUIV)
@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        # bare, the path the regex serves
        ("2", "-2"),
        ("-2", "2"),
        ("8", "-8"),
        ("-8", "8"),
        ("3.5", "-3.5"),
        ("-3.5", "3.5"),
        ("100", "-100"),
        # boxed, alone and inside a sentence
        (r"\boxed{2}", r"\boxed{-2}"),
        (r"\boxed{-2}", r"\boxed{2}"),
        (r"The answer is \boxed{2}.", r"The answer is \boxed{-2}."),
        # dollar, alone and inside a sentence
        (r"$2$", r"$-2$"),
        (r"$-2$", r"$2$"),
        (r"The answer is $2$.", r"The answer is $-2$."),
        # the `=` branch, which does not touch the regex
        ("x = 2", "x = -2"),
        ("x = -2", "x = 2"),
    ],
)
def test_opposite_sign_does_not_match(name, candidate, gold):
    """A sign flip is a different answer, whichever path read it."""
    assert IS_EQUIV[name](candidate, gold) is False


@pytest.mark.parametrize("name", IS_EQUIV)
@pytest.mark.parametrize(
    "answer",
    [
        # bare
        "-2",
        "-8",
        "-3.5",
        "-100",
        "2",
        "8",
        "3.5",
        # boxed, alone and inside a sentence
        r"\boxed{-2}",
        r"The answer is \boxed{-2}.",
        # dollar, alone and inside a sentence
        r"$-2$",
        r"The answer is $-2$.",
        # the `=` branch
        "x = -2",
    ],
)
def test_answer_matches_itself(name, answer):
    """Teaching the regex a sign must not cost the answers that already worked."""
    assert IS_EQUIV[name](answer, answer) is True


@pytest.mark.parametrize("name", IS_EQUIV)
@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        ("2", r"\boxed{2}"),
        ("2", r"$2$"),
        ("2", "x = 2"),
        ("2", r"The answer is \boxed{2}."),
        (r"\boxed{2}", r"$2$"),
        (r"\boxed{2}", "x = 2"),
        (r"$2$", "x = 2"),
        ("-3.5", r"\boxed{-3.5}"),
        ("-3.5", r"$-3.5$"),
        (r"\boxed{-3.5}", r"$-3.5$"),
    ],
)
def test_one_value_reached_through_two_paths_matches(name, candidate, gold):
    """Every path has to agree on the value it read, sign included."""
    assert IS_EQUIV[name](candidate, gold) is True


@pytest.mark.parametrize("name", IS_EQUIV)
@pytest.mark.parametrize(
    "gold",
    [
        # the negative golds in the first 100 rows of hails/agieval-math
        "-1",
        "-6",
        "-21",
        "-15",
        # the negative golds in the first 100 rows of HAERAE-HUB/HRM8K KSM
        r"-8\pi",
        r"-2\pi i",
    ],
)
def test_a_real_negative_gold_matches_itself_and_rejects_the_positive(name, gold):
    """The values the datasets actually hold, not just synthetic ones."""
    assert IS_EQUIV[name](gold, gold) is True
    assert IS_EQUIV[name](gold.lstrip("-"), gold) is False


@pytest.mark.parametrize("name", IS_EQUIV)
@pytest.mark.parametrize(
    ("candidate", "gold"),
    [
        ("2", "-2"),
        ("-2", "2"),
        ("-8", "8"),
        (r"\boxed{3.5}", r"\boxed{-3.5}"),
    ],
)
def test_the_verdict_is_symmetric(name, candidate, gold):
    """Swapping the two answers may not change the verdict."""
    assert IS_EQUIV[name](candidate, gold) is IS_EQUIV[name](gold, candidate)


@pytest.mark.parametrize("name", IS_EQUIV)
@pytest.mark.parametrize(
    ("candidate", "gold", "expected"),
    [
        # unchanged: unsigned answers paired with unsigned answers
        ("2", "2", True),
        ("30", "30", True),
        ("3.5", "3.5", True),
        ("2", "3", False),
        ("2", "30", False),
        # unchanged: signed answers paired with signed answers
        ("-2", "-2", True),
        ("-2", "-3", False),
    ],
)
def test_same_sign_comparisons_are_unchanged(name, candidate, gold, expected):
    """Only the sign flip changes; every other verdict stays as it was."""
    assert IS_EQUIV[name](candidate, gold) is expected
