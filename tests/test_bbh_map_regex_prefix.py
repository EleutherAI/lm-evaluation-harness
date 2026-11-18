"""Regression coverage for BBH mapped regex choice extraction."""

import pytest

from lm_eval.tasks.bbh.cot_zeroshot.utils import MapRegexFilter as CotMapRegexFilter
from lm_eval.tasks.bbh.zeroshot.utils import MapRegexFilter as ZeroMapRegexFilter


@pytest.mark.parametrize("filter_cls", [ZeroMapRegexFilter, CotMapRegexFilter])
def test_map_regex_longer_choice_not_shadowed_by_prefix(filter_cls):
    filt = filter_cls(regex_pattern_to_value={"cat": "short", "caterpillar": "long"})

    assert filt.apply([["caterpillar", "cat", "unrelated"]], [{}]) == [
        ["long", "short", "[invalid]"]
    ]


@pytest.mark.parametrize("filter_cls", [ZeroMapRegexFilter, CotMapRegexFilter])
def test_map_regex_last_match_preserves_group_select(filter_cls):
    filt = filter_cls(
        regex_pattern_to_value={"cat": "short", "caterpillar": "long"},
        group_select=-1,
    )

    assert filt.apply([["cat followed by caterpillar"]], [{}]) == [["long"]]


@pytest.mark.parametrize("filter_cls", [ZeroMapRegexFilter, CotMapRegexFilter])
def test_map_regex_bbh_word_boundary_patterns_still_work(filter_cls):
    filt = filter_cls(
        regex_pattern_to_value={
            r"\b(no|does not tell the truth|is not telling the truth)\b": "no",
            r"\b(yes|tells the truth|is telling the truth)\b": "yes",
        },
        ignore_case=True,
    )

    assert filt.apply([["IS TELLING THE TRUTH", "does not tell the truth"]], [{}]) == [
        ["yes", "no"]
    ]
