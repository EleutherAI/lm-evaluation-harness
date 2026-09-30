import importlib.util
from pathlib import Path

import pytest

from lm_eval.filters.extraction import MultiChoiceRegexFilter, RegexFilter
from lm_eval.filters.transformation import SPANFilter


# The `strict-match` filter of every flan-cot / bbh cot_zeroshot task, e.g.
# lm_eval/tasks/mmlu/flan_cot_zeroshot/_mmlu_flan_cot_zeroshot_template_yaml.
FLAN_STRICT_MATCH = (
    r"((?<=The answer is )(.*)(?=.)|(?<=answer is )(.*)(?=.)"
    r"|(?<=The answer: )(.*)(?=.)|(?<=The final answer: )(.*)(?=.))"
)


def test_regex_whitespace_only_capture_uses_the_fallback():
    # The cue is present but nothing follows it, so the capture holds only
    # whitespace. That is not an answer, and the filter must return the
    # configured fallback for it rather than a bare empty string. Regression:
    # `if match:` tested truthiness before `.strip()`, so "   " passed and the
    # strip then reduced it to "", silently ignoring `fallback`.
    filt = RegexFilter(regex_pattern=FLAN_STRICT_MATCH)

    resps = [["The answer is   \n"]]
    docs = [{"choices": ["(A)", "(B)"]}]

    assert filt.apply(resps, docs) == [["[invalid]"]]


@pytest.mark.parametrize("response", ["The answer is \t ", "The answer is  \t"])
def test_regex_whitespace_only_capture_uses_a_custom_fallback(response):
    # The same response with a non-default fallback, to show the configured
    # value is what gets returned.
    filt = RegexFilter(regex_pattern=FLAN_STRICT_MATCH, fallback="NO_ANSWER")

    assert filt.apply([[response]], [{"choices": ["(A)", "(B)"]}]) == [["NO_ANSWER"]]


def test_regex_keeps_matching_a_response_whose_cue_is_followed_by_an_answer():
    # Only a capture that is empty *after* stripping falls back. A real answer
    # is still returned, and the strip keeps trimming the surrounding padding.
    filt = RegexFilter(regex_pattern=FLAN_STRICT_MATCH)

    assert filt.apply([["The answer is (A)."]], [{"choices": ["(A)"]}]) == [["(A)"]]
    assert filt.apply([["The answer is  (A). "]], [{"choices": ["(A)"]}]) == [["(A)."]]


def test_regex_missing_cue_still_uses_the_fallback():
    # The unchanged non-match path, as a control for the case above.
    filt = RegexFilter(regex_pattern=FLAN_STRICT_MATCH)

    assert filt.apply([["I am not sure"]], [{"choices": ["(A)"]}]) == [["[invalid]"]]


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
