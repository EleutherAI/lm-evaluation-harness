import re

import pytest

from lm_eval.tasks.bbh.cot_zeroshot.utils import (
    ExtendedRegexFilter as CotZeroshotRegexFilter,
    MapRegexFilter as CotZeroshotMapFilter,
)
from lm_eval.tasks.bbh.zeroshot.utils import (
    ExtendedRegexFilter as ZeroshotRegexFilter,
    MapRegexFilter as ZeroshotMapFilter,
)


# `bbh/zeroshot/utils.py` and `bbh/cot_zeroshot/utils.py` are byte-identical
# copies of the same filters, so every case below runs against both.
REGEX_FILTERS = [
    pytest.param(ZeroshotRegexFilter, id="zeroshot"),
    pytest.param(CotZeroshotRegexFilter, id="cot_zeroshot"),
]
MAP_FILTERS = [
    pytest.param(ZeroshotMapFilter, id="zeroshot"),
    pytest.param(CotZeroshotMapFilter, id="cot_zeroshot"),
]

# Both groups are optional and the matched text ("x") lies outside them, so
# `findall` yields a tuple whose every element is the empty string.
OPTIONAL_GROUPS = re.compile(r"x(foo)?(bar)?")


@pytest.mark.parametrize("filter_cls", REGEX_FILTERS)
def test_all_empty_tuple_returns_empty_string(filter_cls):
    """An all-empty capture tuple is a non-match, not an IndexError.

    `[m for m in match if m][0]` indexed an empty list here, so a regex with two
    optional groups crashed the run instead of falling through to the fallback.
    Mirrors the guard in lm_eval/filters/extraction.py.
    """
    assert OPTIONAL_GROUPS.findall("x") == [("", "")]
    assert filter_cls().find_match(OPTIONAL_GROUPS, "x") == ""


@pytest.mark.parametrize("filter_cls", REGEX_FILTERS)
def test_partially_empty_tuple_takes_first_non_empty_group(filter_cls):
    """The surviving group is still selected when only some groups are empty."""
    assert OPTIONAL_GROUPS.findall("xbar") == [("", "bar")]
    assert filter_cls().find_match(OPTIONAL_GROUPS, "xbar") == "bar"


@pytest.mark.parametrize("filter_cls", REGEX_FILTERS)
def test_single_group_match_is_unchanged(filter_cls):
    """The common path -- one capture group, so `findall` returns plain strings."""
    regex = re.compile(r"\(([A-Z])\)")
    assert filter_cls().find_match(regex, "The answer is (B)") == "B"


@pytest.mark.parametrize("filter_cls", REGEX_FILTERS)
def test_convert_dict_still_applied(filter_cls):
    """A matched value is still mapped through `convert_dict`."""
    regex = re.compile(r"\(([A-Z])\)")
    assert filter_cls().find_match(regex, "(B)", {"B": "(B)"}) == "(B)"


@pytest.mark.parametrize("filter_cls", MAP_FILTERS)
def test_map_filter_falls_back_instead_of_crashing(filter_cls):
    """End to end: an all-empty tuple yields the fallback, not a crash.

    MapRegexFilter joins its configured patterns with "|", which is how a
    multi-group regex reaches `find_match` in the first place.
    """
    filter_instance = filter_cls(
        regex_pattern_to_value={r"x(foo)?": "X", r"y(bar)?": "Y"}
    )
    assert filter_instance.apply([["x"]], [{}]) == [["[invalid]"]]
