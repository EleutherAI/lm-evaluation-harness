import re

import pytest

from lm_eval.tasks.mmlu.flan_cot_zeroshot.utils import (
    MultiChoiceRegexFilter as CotZeroshotRegexFilter,
)
from lm_eval.tasks.mmlu.flan_n_shot.generative.utils import (
    MultiChoiceRegexFilter as GenerativeRegexFilter,
)


# `mmlu/flan_cot_zeroshot/utils.py` and `mmlu/flan_n_shot/generative/utils.py`
# carry the same `MultiChoiceRegexFilter`, so every case below runs against both.
REGEX_FILTERS = [
    pytest.param(GenerativeRegexFilter, id="flan_n_shot_generative"),
    pytest.param(CotZeroshotRegexFilter, id="flan_cot_zeroshot"),
]

# Both groups are optional and the matched text ("x") lies outside them, so
# `findall` yields a tuple whose every element is the empty string.
OPTIONAL_GROUPS_PATTERN = r"x(foo)?(bar)?"


@pytest.mark.parametrize("filter_cls", REGEX_FILTERS)
def test_all_empty_tuple_falls_back(filter_cls):
    """An all-empty capture tuple is a non-match, not an IndexError.

    `[m for m in match if m][0]` indexed an empty list here, so a regex with two
    optional groups crashed the run instead of falling through to the fallback.
    This mirrors the guard already applied to the bbh filters (see
    tests/test_bbh_filters.py) and to lm_eval/filters/extraction.py.
    """
    assert re.compile(OPTIONAL_GROUPS_PATTERN).findall("x") == [("", "")]
    filter_instance = filter_cls(regex_pattern=OPTIONAL_GROUPS_PATTERN)
    resps = [["x"]]
    docs = [{"choices": ["A", "B"]}]
    filtered = filter_instance.apply(resps, docs)
    assert filtered == [["[invalid]"]]


@pytest.mark.parametrize("filter_cls", REGEX_FILTERS)
def test_populated_capture_group_still_extracted(filter_cls):
    """A non-empty capture group is still returned, so the guard is not over-broad."""
    filter_instance = filter_cls(regex_pattern=r"answer is \(([A-D])\)")
    resps = [["The answer is (B)."]]
    docs = [{"choices": ["A", "B", "C", "D"]}]
    filtered = filter_instance.apply(resps, docs)
    assert filtered == [["B"]]
