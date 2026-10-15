import pytest


# `minerva_math.utils` imports math_verify and antlr4-python3-runtime at module
# scope and raises without them. Those live in the [math] extra, which the unit
# test workflow does not install, so this module skips there instead of failing.
minerva_utils = pytest.importorskip(
    "lm_eval.tasks.minerva_math.utils",
    reason="minerva_math needs the [math] extra: "
    "sympy, math_verify, antlr4-python3-runtime==4.11",
)

from lm_eval.tasks.leaderboard.math.utils import (
    normalize_final_answer as norm_leaderboard,
)
from lm_eval.tasks.minerva_math.utils import normalize_final_answer as norm
from lm_eval.tasks.putnam_axiom.utils import normalize_final_answer as norm_putnam


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
    r"""Reachable rather than theoretical: `\text{}` is in REMOVED_EXPRESSIONS, so `normalize_final_answer` strips such an answer to "" before it gets here."""
    assert not is_equiv("", "")


def test_mathematically_equal_answers_still_compare_equal():
    """The sympy path is still reached for strings that are not identical."""
    assert is_equiv("\\frac{1}{2}", "0.5")


"""The thousands-separator comma strip must not fuse bare digit tuples.

`normalize_final_answer` strips commas when the result is digit-only, which
fuses tuple answers: "0,1" -> "01". Sixteen MATH test-set gold answers are
bare tuples ("0,1" x4, "3,5,7", "12,10,6", ...), and the fused gold can no
longer match a model's "(0,1)" or "0, 1" spelling.
"""

NORMS = (norm, norm_leaderboard, norm_putnam)


def test_bare_digit_tuple_is_not_fused():
    for f in NORMS:
        assert f("0,1") == "0,1"
        assert f("0, 1") == "0,1"
        assert f("3,5,7") == "3,5,7"
        assert f("12,10,6") == "12,10,6"


def test_true_thousands_separators_are_still_stripped():
    for f in NORMS:
        assert f("100,000") == "100000"
        assert f("115,000") == "115000"
        assert f("1,000") == "1000"
        assert f("61,328") == "61328"


def test_malformed_grouping_is_left_untouched():
    # not valid thousands grouping: keep as-is instead of guessing
    for f in NORMS:
        assert f("12,34") == "12,34"
        assert f("1234,5678") == "1234,5678"


def test_negative_thousands_grouping_is_stripped():
    for f in NORMS:
        assert f("-1,024") == "-1024"


def test_equiv_harm_case_tuple_spelling_variants():
    # same tuple spelled with/without spaces must compare equal via the
    # normalized string path (parse_latex cannot parse bare tuples)
    assert norm("0,1") == norm("0, 1")


r"""Indexed-root survival through normalize_final_answer.

The sqrt shorthand rule used to consume the "[" of an optional root index,
turning e.g. \\sqrt[3]{8} into the invalid \\sqrt{[}3]{8}; see issue #4036.
"""

normalize = minerva_utils.normalize_final_answer


@pytest.mark.parametrize(
    "answer",
    [
        "\\sqrt[3]{8}",
        "\\sqrt[3]{2}+1",
        "\\sqrt[4]{16}",
        # unbraced argument gets canonicalized to the braced form
        ("\\sqrt[3]8", "\\sqrt[3]{8}"),
        # classic shorthands must keep their old behavior byte-for-byte
        ("\\sqrt2", "\\sqrt{2}"),
        ("\\sqrta", "\\sqrt{a}"),
    ],
)
def test_normalize_preserves_indexed_roots(answer):
    expected = answer[1] if isinstance(answer, tuple) else answer
    assert normalize(answer[0] if isinstance(answer, tuple) else answer) == expected


@pytest.mark.parametrize(
    "pred,gold",
    [
        # model writes a cube root, gold is the simplified integer
        ("\\sqrt[3]{8}", "2"),
        # brace vs bare argument spellings of the same root
        ("\\sqrt[3]2", "\\sqrt[3]{2}"),
        # fourth power root vs its integer value
        ("\\sqrt[4]{16}", "2"),
    ],
)
def test_is_equiv_indexed_roots_through_pipeline(pred, gold):
    assert is_equiv(normalize(pred), normalize(gold))


def test_is_equiv_shorthand_still_works():
    assert is_equiv(normalize("\\sqrt8"), normalize("2\\sqrt{2}"))
