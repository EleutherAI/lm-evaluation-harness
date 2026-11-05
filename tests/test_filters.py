import importlib.util
from pathlib import Path

import pytest

from lm_eval.filters.extraction import MultiChoiceRegexFilter
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


def test_format_span_normalizes_label_only():
    # Labels are normalized, but entity text containing label-words as
    # substrings (e.g. "Company", "Country", "George") must be left intact.
    filt = SPANFilter()
    resps = [["ORGANIZATION: Shell Company $ LOCATION: Country Club $ PERSON: George"]]

    assert filt.apply(resps, [{}]) == [
        ["org: shell company $ loc: country club $ per: george"]
    ]


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


def test_multi_choice_regex_prefix_fix_holds_under_task_config():
    # Every task that uses this filter passes group_select=-1 and a "(\\([A-Z]\\))"
    # pattern, not the defaults; the shorter choice named alone must still win.
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [{"choices": ["Guilty", "Guilty of Romance"]}]

    assert filt.apply([["the answer is Guilty of Romance"]], docs) == [["(B)"]]
    assert filt.apply([["the answer is Guilty"]], docs) == [["(A)"]]


def test_multi_choice_regex_colliding_choices_resolved_by_their_original_text():
    # Distinct choices can share one normalized text once case and punctuation
    # are ignored ("-5"/"5" -> "5", "1.5"/"15" -> "15", "Na"/"NA" -> "na"); each
    # choice keeps its own letter and the answer follows the original spelling
    # the response used.
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )

    for choices, response, want in [
        (["-5", "5", "10", "-10"], "So x = -5.", "(A)"),
        (["-5", "5", "10", "-10"], "So x = 5.", "(B)"),
        (["15", "1.5", "2", "3"], "The value is 1.5.", "(B)"),
        (["Na", "NA", "K", "Li"], "The element is Na", "(A)"),
    ]:
        assert filt.apply([[response]], [{"choices": choices}]) == [[want]], (choices, response)


def test_multi_choice_regex_collision_picks_the_same_group_select_match():
    # The resolution may only consider the match that group_select selects: an
    # earlier colliding choice named between other text must not flip the
    # answer to its letter.
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )

    assert (
        filt.apply(
            [["Not -5; the answer is 5."]],
            [{"choices": ["-5", "5", "10", "-10"]}],
        )
        == [["(B)"]]
    )


def test_multi_choice_regex_exact_original_spelling_wins_before_tie():
    # Both spellings normalize to the same text; the response reusing a choice's
    # exact spelling resolves to that choice, and a response that matches
    # neither spelling resolves to the group's earliest choice.
    filt = MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=-1,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [{"choices": ["Na", "NA", "K", "Li"]}]

    assert filt.apply([["The element is NA"]], docs) == [["(B)"]]
    assert filt.apply([["the element is na"]], docs) == [["(A)"]]


@pytest.mark.parametrize("variant", ["zeroshot", "cot_zeroshot"])
def test_bbh_multi_choice_regex_prefix_choice_does_not_shadow_longer_choice(variant):
    # bbh keeps its own copy of MultiChoiceRegexFilter, wired via
    # `!function utils.MultiChoiceRegexFilter`, so it needs the same longest-first
    # ordering. Mirrors the bbh_zeroshot_movie_recommendation filter config.
    path = Path(__file__).parent.parent / f"lm_eval/tasks/bbh/{variant}/utils.py"
    spec = importlib.util.spec_from_file_location(f"bbh_{variant}_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    filt = utils.MultiChoiceRegexFilter(
        regex_pattern=r"(\([A-Z]\))",
        group_select=0,
        ignore_case=True,
        ignore_punctuation=True,
    )
    docs = [
        {
            "input": "Find a movie similar to Batman Begins:\nOptions:\n"
            "(A) Batman\n(B) Batman Returns\n(C) Alien\n(D) Titanic"
        }
    ]

    assert filt.apply([["The answer is Batman Returns"]], docs) == [["(B)"]]
    assert filt.apply([["The answer is Batman"]], docs) == [["(A)"]]
