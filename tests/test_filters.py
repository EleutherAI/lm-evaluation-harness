from lm_eval.filters.extraction import MultiChoiceRegexFilter, RegexFilter
from lm_eval.filters.transformation import SPANFilter



def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_choice_text():
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["alpha"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(A)"]]


def test_multi_choice_regex_all_empty_capture_groups_falls_back_to_bare_letter():
    filt = MultiChoiceRegexFilter(regex_pattern=r"()()")

    resps = [[": B"]]
    docs = [{"choices": ["alpha", "beta"]}]

    assert filt.apply(resps, docs) == [["(B)"]]



def test_regex_filter_group_select_out_of_range_falls_back():
    # only one "#### <num>" match exists, but we ask for the 2nd -> must not crash
    filt = RegexFilter(group_select=1)
    resps = [["The result is #### 42"]]
    docs = [{}]

    assert filt.apply(resps, docs) == [["[invalid]"]]


def test_regex_filter_negative_group_select_takes_last_match():
    filt = RegexFilter(group_select=-1)
    resps = [["#### 1 then #### 2"]]
    docs = [{}]

    assert filt.apply(resps, docs) == [["2"]]


def test_regex_filter_default_group_select_unchanged():
    filt = RegexFilter()
    resps = [["#### 7"]]
    docs = [{}]

    assert filt.apply(resps, docs) == [["7"]]

def test_format_span_normalizes_label_only():
    # Labels are normalized, but entity text containing label-words as
    # substrings (e.g. "Company", "Country", "George") must be left intact.
    filt = SPANFilter()
    resps = [["ORGANIZATION: Shell Company $ LOCATION: Country Club $ PERSON: George"]]

    assert filt.apply(resps, [{}]) == [
        ["org: shell company $ loc: country club $ per: george"]
    ]

