import pytest

from lm_eval.tasks.leaderboard.math.utils import (
    normalize_final_answer as normalize_leaderboard,
)
from lm_eval.tasks.minerva_math.utils import (
    normalize_final_answer as normalize_minerva,
)
from lm_eval.tasks.putnam_axiom.utils import (
    normalize_final_answer as normalize_putnam,
)


NORMALIZERS = [normalize_minerva, normalize_leaderboard, normalize_putnam]


@pytest.mark.parametrize("normalize_final_answer", NORMALIZERS)
def test_latex_commands_survive_unit_removal(normalize_final_answer):
    # https://github.com/EleutherAI/lm-evaluation-harness/issues/4031
    # unit words used to be substring-removed from command names:
    # \left -> \le, \infty -> \iny, \square -> \, corrupting valid LaTeX
    assert normalize_final_answer(r"\left(1,2\right)") == r"\left(1,2\right)"
    assert normalize_final_answer(r"\infty") == r"\infty"
    assert normalize_final_answer(r"\square") == r"\square"
    assert normalize_final_answer(r"5\text{cm}") == "5"


@pytest.mark.parametrize("normalize_final_answer", NORMALIZERS)
def test_unit_after_command_still_stripped(normalize_final_answer):
    # whitespace between a command and a unit is removed by SUBSTITUTIONS;
    # the unit must be stripped while whitespace still delimits it
    # (regression found in review: maximal-munch protection kept "cm"
    # inside "\picm", flipping is_equiv("2\pi cm", "2\pi") to False)
    assert normalize_final_answer(r"2\pi cm") == r"2\pi"
    assert normalize_final_answer(r"30^\circ inches") == "30"


@pytest.mark.parametrize("normalize_final_answer", NORMALIZERS)
def test_unit_words_still_removed(normalize_final_answer):
    # units glued to numbers (spaces are stripped before unit removal)
    assert normalize_final_answer("12 ft") == "12"
    assert normalize_final_answer("12ft") == "12"
    assert normalize_final_answer(r"5 \text{cm}") == "5"
    assert normalize_final_answer("2 dollars") == "2"


@pytest.mark.parametrize("normalize_final_answer", NORMALIZERS)
def test_symbolic_removals_unaffected(normalize_final_answer):
    assert normalize_final_answer("3^\\circ") == "3"
    assert normalize_final_answer("{,}1,2") == "12"


@pytest.mark.parametrize("normalize_final_answer", NORMALIZERS)
def test_zero_separator_glued_command_kept(normalize_final_answer):
    # known trade-off: literal "<command><unit>" with no separator or braces
    # is treated as one token (pristine main stripped it by substring luck;
    # corpus scan found zero such shapes among hendrycks_math golds)
    assert normalize_final_answer(r"2\picm") == r"2\picm"


def test_is_equiv_end_to_end():
    # the actual contract: corrupted normalization made these score 0;
    # process_results normalizes both sides before calling is_equiv
    pytest.importorskip("sympy")
    pytest.importorskip("math_verify")
    from lm_eval.tasks.minerva_math.utils import is_equiv, normalize_final_answer

    assert is_equiv(
        normalize_final_answer(r"\left(\frac{3}{2}\right)"),
        normalize_final_answer(r"\frac{3}{2}"),
    )
    assert is_equiv(normalize_final_answer(r"2\pi cm"), normalize_final_answer(r"2\pi"))
    assert is_equiv(
        normalize_final_answer(r"30^\circ inches"), normalize_final_answer(r"30")
    )
