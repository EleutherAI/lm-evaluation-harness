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


def test_pos_filter_returns_list_not_generator():
    """POSFilter.apply must return a concrete list, like every other filter.

    Filter.apply is documented to return a list of per-instance results, and
    FilterEnsemble chains filters by feeding one filter's output as the next
    filter's input. POSFilter previously returned a generator, which violates
    that contract and is silently consumed if the results are iterated twice.
    """
    from lm_eval.filters.extraction import POSFilter

    f = POSFilter()
    resps = [["[('the', 'DET'), ('cat', 'NOUN')]"]]
    out = f.apply(resps, [{}])

    assert isinstance(out, list)
    assert out == [[["DET", "NOUN"]]]
    # A generator would be empty on the second pass; a list is stable.
    assert list(out) == list(out)


def test_pos_filter_falls_back_when_no_tags():
    from lm_eval.filters.extraction import POSFilter

    f = POSFilter()
    out = f.apply([["no tagged tokens here"]], [{}])
    assert out == [[["invalid"]]]
