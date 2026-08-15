from lm_eval.filters.extraction import MultiChoiceRegexFilter


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


def test_multi_choice_regex_prefix_choice_does_not_shadow_longer_choice():
    # When one choice's text is a prefix of another, naming the longer choice in the
    # response must map to the longer choice's letter. Regression: the fallback regex
    # joined choices in list order, and leftmost-alternation let the shorter prefix
    # ("Guilty") shadow "Guilty of Romance", returning (A) instead of (B).
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"()()",
        ignore_case=True,
        ignore_punctuation=True,
    )

    resps = [["the answer is Guilty of Romance"]]
    docs = [{"choices": ["Guilty", "Guilty of Romance"]}]

    assert filt.apply(resps, docs) == [["(B)"]]
