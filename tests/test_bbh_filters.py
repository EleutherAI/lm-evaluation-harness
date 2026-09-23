import re

import pytest

from lm_eval.tasks.bbh.cot_zeroshot.utils import (
    ExtendedRegexFilter as CotZeroshotRegexFilter,
    MapRegexFilter as CotZeroshotMapFilter,
    NumberParseRegexFilter as CotZeroshotNumberFilter,
)
from lm_eval.tasks.bbh.zeroshot.utils import (
    ExtendedRegexFilter as ZeroshotRegexFilter,
    MapRegexFilter as ZeroshotMapFilter,
    NumberParseRegexFilter as ZeroshotNumberFilter,
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
NUMBER_FILTERS = [
    pytest.param(ZeroshotNumberFilter, id="zeroshot"),
    pytest.param(CotZeroshotNumberFilter, id="cot_zeroshot"),
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


def parse_number(filter_cls, resp, group_select=0):
    # Same regex_pattern as the object_counting and multistep_arithmetic_two configs.
    filter_instance = filter_cls(regex_pattern="([-0-9]+)", group_select=group_select)
    return filter_instance.apply([[resp]], [{}])[0][0]


@pytest.mark.parametrize("filter_cls", NUMBER_FILTERS)
@pytest.mark.parametrize(
    "resp",
    [
        "I cannot help with this; it is often unclear.",
        "Please pay attention to the question.",
        "Ask someone else.",
        "There are none.",
    ],
)
def test_number_word_inside_another_word_is_not_a_number(filter_cls, resp):
    """Used to give 10 for "often" and "attention", and 1 for "someone" and "none"."""
    assert parse_number(filter_cls, resp) == "[invalid]"


@pytest.mark.parametrize("filter_cls", NUMBER_FILTERS)
@pytest.mark.parametrize(
    "resp, expected",
    [
        ("six", "6"),
        ("There are seven objects.", "7"),
        ("Eight", "8"),
        ("The answer is nine.", "9"),
        ("ninety nine", "99"),
        ("twenty eight thousand", "28000"),
    ],
)
def test_bare_six_to_nine_parse(filter_cls, resp, expected):
    """Six, seven, eight and nine used to need a "teen"/"ty"/"een"/"y" suffix."""
    assert parse_number(filter_cls, resp) == expected


@pytest.mark.parametrize("filter_cls", NUMBER_FILTERS)
@pytest.mark.parametrize(
    "resp, expected",
    [
        ("I looked for it.", "[invalid]"),
        ("fourteen", "14"),
        ("forty", "40"),
        ("fifty", "50"),
        ("sixteen", "16"),
        ("sixty", "60"),
        ("seventy", "70"),
        ("eighty", "80"),
        ("ninety", "90"),
        ("twenty one", "21"),
        ("one hundred", "100"),
        ("three hundred and twenty", "320"),
        ("two thousand three hundred", "2300"),
    ],
)
def test_number_words_parse(filter_cls, resp, expected):
    """Teens, tens and multi-word numbers parse; the word "for" does not."""
    assert parse_number(filter_cls, resp) == expected


@pytest.mark.parametrize("filter_cls", NUMBER_FILTERS)
@pytest.mark.parametrize(
    "resp",
    [
        "million billion",
        "two million and one thousand thousand",
        "a thousand and one",
    ],
)
def test_unparseable_number_words_fall_back(filter_cls, resp):
    """Word runs that word2number rejects give the fallback, not an exception."""
    assert parse_number(filter_cls, resp) == "[invalid]"


@pytest.mark.parametrize("filter_cls", NUMBER_FILTERS)
def test_cot_config_takes_last_number_word(filter_cls):
    """The cot_zeroshot configs use group_select: -1, so the last number wins."""
    resp = "Let me count: one, two, three... there are seven objects."
    assert parse_number(filter_cls, resp, group_select=-1) == "7"
