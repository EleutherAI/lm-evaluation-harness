import re

import pytest

from lm_eval.tasks.mmlu.flan_cot_zeroshot.utils import (
    MultiChoiceRegexFilter as CotZeroshotFilter,
)
from lm_eval.tasks.mmlu.flan_n_shot.generative.utils import (
    MultiChoiceRegexFilter as GenerativeFilter,
)


# `mmlu/flan_cot_zeroshot/utils.py` and `mmlu/flan_n_shot/generative/utils.py`
# are byte-identical copies of the same filter, so every case runs against both.
FILTERS = [
    pytest.param(CotZeroshotFilter, id="flan_cot_zeroshot"),
    pytest.param(GenerativeFilter, id="flan_n_shot_generative"),
]

DOC = {"choices": ["red", "green", "blue", "yellow"]}

# Both groups are optional and the matched text ("x") lies outside them, so
# `findall` yields a tuple whose every element is the empty string.
OPTIONAL_GROUPS = r"x(foo)?(bar)?"


@pytest.mark.parametrize("filter_cls", FILTERS)
def test_all_empty_tuple_falls_back_instead_of_crashing(filter_cls):
    """An all-empty capture tuple is a non-match, not an IndexError.

    `[m for m in match if m][0]` indexed an empty list here. Mirrors the guard
    in lm_eval/filters/extraction.py and lm_eval/tasks/bbh/*/utils.py.
    """
    assert re.findall(OPTIONAL_GROUPS, "x") == [("", "")]
    f = filter_cls(regex_pattern=OPTIONAL_GROUPS)
    assert f.apply([["x"]], [DOC]) == [["[invalid]"]]


@pytest.mark.parametrize("filter_cls", FILTERS)
def test_all_empty_tuple_reaches_choice_fallback(filter_cls):
    """After the non-match, the choice-text fallback still runs."""
    f = filter_cls(regex_pattern=OPTIONAL_GROUPS)
    assert f.apply([["x, so it is blue"]], [DOC]) == [["(C)"]]


@pytest.mark.parametrize("filter_cls", FILTERS)
def test_partially_empty_tuple_takes_first_non_empty_group(filter_cls):
    """The surviving group is still selected when only some groups are empty."""
    f = filter_cls(regex_pattern=OPTIONAL_GROUPS)
    assert f.apply([["xbar"]], [DOC]) == [["bar"]]


@pytest.mark.parametrize("filter_cls", FILTERS)
def test_shipped_config_is_unchanged(filter_cls):
    """The configured single-group pattern behaves as before."""
    f = filter_cls(regex_pattern=r"(\([A-Z]\))", ignore_case=True)
    assert f.apply([["The answer is (B)."]], [DOC]) == [["(B)"]]
    assert f.apply([["I pick: D"]], [DOC]) == [["(D)"]]
