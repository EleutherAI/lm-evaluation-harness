import importlib.util
from pathlib import Path

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_filter
from lm_eval.filters import build_filter_ensemble
from lm_eval.filters.custom import CustomFilter
from lm_eval.filters.extraction import MultiChoiceRegexFilter
from lm_eval.filters.selection import (
    MajorityVoteFilter,
    TakeFirstFilter,
    TakeKFilter,
)
from lm_eval.filters.transformation import (
    LowercaseFilter,
    MapFilter,
    SPANFilter,
    UppercaseFilter,
)


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


def test_take_first_returns_first_response_per_doc():
    filt = TakeFirstFilter()

    assert list(filt.apply([["a", "b"], ["c"]], [{}] * 2)) == ["a", "c"]


def test_take_first_k_keeps_first_k_and_validates_repeats():
    filt = TakeKFilter(k=2)

    assert list(filt.apply([["a", "b", "c"], ["d", "e", "f"]], [{}] * 2)) == [
        ["a", "b"],
        ["d", "e"],
    ]

    with pytest.raises(AssertionError, match="repeats"):
        TakeKFilter(k=3).apply([["a", "b"]], [{}])


def test_majority_vote_plurality_breaks_ties_by_first_occurrence():
    filt = MajorityVoteFilter()

    assert list(filt.apply([["a", "a", "b"], ["b", "a", "b"]], [{}] * 2)) == [
        ["a"],
        ["b"],
    ]
    # duplicate counts are tied, first occurrence in the response list wins
    assert list(filt.apply([["a", "b"]], [{}])) == [["a"]]


@pytest.mark.parametrize(
    ("resps", "expected"),
    [
        (["HeLLo", "WORLD"], ["hello", "world"]),
        (["ÄBC"], ["äbc"]),
    ],
)
def test_lowercase_filter_applies_str_lower(resps, expected):
    assert LowercaseFilter().apply([resps], [{}]) == [expected]


@pytest.mark.parametrize(
    ("resps", "expected"),
    [
        (["hello", "world"], ["HELLO", "WORLD"]),
        (["straße"], ["STRASSE"]),
    ],
)
def test_uppercase_filter_applies_str_upper(resps, expected):
    assert UppercaseFilter().apply([resps], [{}]) == [expected]


def test_map_filter_maps_known_keys_and_defaults_others():
    filt = MapFilter({"yes": 1, "no": 0}, default_value=-1)

    assert filt.apply([["yes", "maybe", "no", None]], [{}]) == [[1, -1, 0, -1]]
    assert MapFilter().apply([["anything"]], [{}]) == [[None]]

    with pytest.raises(AssertionError, match="mapping_dict"):
        MapFilter(mapping_dict=["not", "a", "dict"])


def test_custom_filter_applies_user_function():
    def exclaim(resps, docs):
        return [[f"{r}!" for r in doc_resps] for doc_resps in resps]

    assert CustomFilter(filter_fn=exclaim).apply([["hi", "yo"]], [{}]) == [
        ["hi!", "yo!"]
    ]


@pytest.mark.parametrize(
    "name",
    [
        "take_first",
        "take_first_k",
        "majority_vote",
        "lowercase",
        "uppercase",
        "map",
        "custom",
    ],
)
def test_registered_filter_names_resolve_via_registry(name):
    # task configs reference these filters by name as strings; they must resolve
    # to a callable so `build_filter_ensemble` can instantiate them
    assert callable(get_filter(name))


def _make_instances(docs_and_resps):
    return [
        Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(str(doc["i"]),),
            idx=doc["i"],
            resps=resps,
        )
        for doc, resps in docs_and_resps
    ]


def test_filter_ensemble_chains_filters_and_stores_per_instance():
    instances = _make_instances(
        [({"i": 0}, ["First", "second"]), ({"i": 1}, ["Third", "fourth"])]
    )

    build_filter_ensemble(
        "normalize", [("lowercase", None), ("take_first", None)]
    ).apply(instances)

    assert [inst.filtered_resps["normalize"] for inst in instances] == [
        "first",
        "third",
    ]


def test_filter_ensemble_passes_filter_kwargs():
    instances = _make_instances([({"i": 0}, ["yes", "no", "YES"])])

    build_filter_ensemble(
        "mapped",
        [
            ("lowercase", None),
            ("map", {"mapping_dict": {"yes": 1, "no": 0}, "default_value": -1}),
        ],
    ).apply(instances)

    assert instances[0].filtered_resps["mapped"] == [1, 0, 1]


def test_filter_ensemble_supports_multiple_named_pipelines():
    instances = _make_instances([({"i": 0}, ["a", "b"])])

    build_filter_ensemble("keep_one", [("take_first_k", {"k": 1})]).apply(instances)
    build_filter_ensemble("keep_two", [("take_first_k", {"k": 2})]).apply(instances)

    assert instances[0].filtered_resps["keep_one"] == ["a"]
    assert instances[0].filtered_resps["keep_two"] == ["a", "b"]
