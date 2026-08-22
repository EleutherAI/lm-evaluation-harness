"""The thousands-separator comma strip must not fuse bare digit tuples.

`normalize_final_answer` strips commas when the result is digit-only, which
fuses tuple answers: "0,1" -> "01". Sixteen MATH test-set gold answers are
bare tuples ("0,1" x4, "3,5,7", "12,10,6", ...), and the fused gold can no
longer match a model's "(0,1)" or "0, 1" spelling.
"""

from lm_eval.tasks.leaderboard.math.utils import (
    normalize_final_answer as norm_leaderboard,
)
from lm_eval.tasks.minerva_math.utils import normalize_final_answer as norm
from lm_eval.tasks.putnam_axiom.utils import (
    normalize_final_answer as norm_putnam,
)


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
