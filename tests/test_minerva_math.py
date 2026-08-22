import pytest


# `minerva_math.utils` imports math_verify and antlr4-python3-runtime at module
# scope and raises without them. Those live in the [math] extra, which the unit
# test workflow does not install, so this module skips there instead of failing.
minerva_utils = pytest.importorskip(
    "lm_eval.tasks.minerva_math.utils",
    reason="minerva_math needs the [math] extra: "
    "sympy, math_verify, antlr4-python3-runtime==4.11",
)

is_equiv = minerva_utils.is_equiv


@pytest.mark.parametrize(
    "answer",
    [
        # sympy's LaTeX parser raises on all of these, so before the fix
        # `is_equiv(answer, answer)` was False and the task scored a
        # character-for-character correct answer as wrong.
        pytest.param("(2,4)", id="tuple"),
        pytest.param("(-1,2,3)", id="triple"),
        pytest.param("[2,5)", id="half-open-interval"),
        pytest.param("(-\\infty,3]", id="unbounded-interval"),
        pytest.param("\\begin{pmatrix}1\\\\2\\end{pmatrix}", id="column-matrix"),
    ],
)
def test_unparseable_answer_matches_itself(answer):
    """An answer identical to the gold is correct even if sympy cannot parse it."""
    assert is_equiv(answer, answer)


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("2", id="integer"),
        pytest.param("\\frac{1}{2}", id="fraction"),
        pytest.param("x+1", id="expression"),
        # sympy parses this one, so it reached the comparison below unaided --
        # it is here rather than above because the test names should say which
        # path an input actually takes.
        pytest.param("\\text{east}", id="text-answer"),
    ],
)
def test_parseable_answer_still_matches_itself(answer):
    """The cases that already worked keep working."""
    assert is_equiv(answer, answer)


@pytest.mark.parametrize(
    "x1, x2",
    [
        # Different answers of a kind sympy cannot parse must stay unequal --
        # the shortcut is exact-match only and must not collapse these.
        pytest.param("(2,4)", "(3,5)", id="different-tuples"),
        pytest.param("[2,5)", "[2,6)", id="different-intervals"),
        pytest.param("(2,4)", "(4,2)", id="reordered-tuple"),
        pytest.param("\\text{east}", "\\text{west}", id="different-text"),
    ],
)
def test_different_answers_are_not_equivalent(x1, x2):
    assert not is_equiv(x1, x2)


def test_two_empty_answers_are_not_a_match():
    r"""Reachable rather than theoretical: `\text{}` is in REMOVED_EXPRESSIONS,
    so `normalize_final_answer` strips such an answer to "" before it gets here.
    """
    assert not is_equiv("", "")


def test_mathematically_equal_answers_still_compare_equal():
    """The sympy path is still reached for strings that are not identical."""
    assert is_equiv("\\frac{1}{2}", "0.5")
