"""Preserve regex selection semantics with bounded match storage."""

import gc
import re
import tracemalloc

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.filters.extraction import MultiChoiceRegexFilter, RegexFilter


CASES = [
    (r"\d+", "12 34 56"),
    (r"#### (\-?[0-9\.\,]+)", "#### -1,250 #### 4.25 #### 99"),
    (r"([A-Z])|(\d+)", "A 12 B 34"),
    (r"(a)?b", "b ab b"),
    (r"()()", "abc"),
    (r"(?=(\d{2}))", "1234"),
    (r"(?m)^(\w+)", "alpha\nβήτα\n猫"),
    (r"(a*)", "a b"),
    (r"猫|狗", "猫 狗 猫"),
    (r"\d+", "no numbers"),
    (r"\d+", None),
    (r"(a)?", 42),
]


def _findall_reference(pattern, response, index, fallback="[invalid]"):
    matches = re.findall(pattern, response if isinstance(response, str) else "")
    if not matches:
        return fallback
    selected = matches[index]
    if isinstance(selected, tuple):
        selected = next((part for part in selected if part), fallback)
    return selected.strip()


@pytest.mark.parametrize("pattern,response", CASES)
@pytest.mark.parametrize("index", [0, 1, 2, -1, -2, -3, 99, -99, 10**100, -(10**100)])
def test_regex_selection_matches_findall(pattern, response, index):
    extractor = RegexFilter(pattern, group_select=index)
    try:
        expected = _findall_reference(pattern, response, index)
    except IndexError:
        with pytest.raises(IndexError):
            extractor.apply([[response]], [{}])
    else:
        assert extractor.apply([[response]], [{}]) == [[expected]]


@pytest.mark.parametrize(
    "index,expected", [(0, "(B)"), (1, "(A)"), (-1, "(C)"), (-2, "(A)")]
)
def test_multiple_choice_primary_capture_selection(index, expected):
    extractor = MultiChoiceRegexFilter(regex_pattern=r"(\([A-C]\))", group_select=index)
    assert extractor.apply(
        [["Reasoning (B), then (A), finally (C)."]],
        [{"choices": ["red", "green", "blue"]}],
    ) == [[expected]]


@pytest.mark.parametrize(
    "index,expected", [(0, "(B)"), (1, "(A)"), (-1, "(C)"), (-2, "(A)")]
)
def test_multiple_choice_text_fallback_selection(index, expected):
    extractor = MultiChoiceRegexFilter(regex_pattern=r"never-match", group_select=index)
    assert extractor.apply(
        [["green red blue"]], [{"choices": ["red", "green", "blue"]}]
    ) == [[expected]]


@pytest.mark.parametrize("pattern", [r"()()", r"(a)?(b)?"])
def test_empty_capture_fallbacks_are_preserved(pattern):
    assert RegexFilter(pattern, fallback="missing").apply([["x"]], [{}]) == [
        ["missing"]
    ]
    assert MultiChoiceRegexFilter(pattern, fallback="missing").apply(
        [["x"]], [{"choices": ["red", "green"]}]
    ) == [["missing"]]


def test_response_groups_and_iterables_keep_their_order():
    responses = iter([["12 34", "56 78"], [], ["90 12"]])
    assert RegexFilter(r"(\d+)", group_select=-1).apply(responses, [{}, {}, {}]) == [
        ["34", "78"],
        [],
        ["12"],
    ]


@pytest.mark.parametrize(
    "index,expected", [(0, "12"), (1, "34"), (-1, "56"), (-2, "34")]
)
def test_native_task_filter_pipeline_keeps_its_score(index, expected):
    dataset = datasets.DatasetDict(
        {
            "test": datasets.Dataset.from_list(
                [{"prompt": "Question", "target": expected}]
            )
        }
    )
    task = ConfigurableTask(
        config={
            "task": "regex-selection-fixture",
            "custom_dataset": lambda **kwargs: dataset,
            "test_split": "test",
            "doc_to_text": "prompt",
            "doc_to_target": "target",
            "output_type": "generate_until",
            "generation_kwargs": {"do_sample": False, "until": ["\n"]},
            "filter_list": [
                {
                    "name": "selected",
                    "filter": [
                        {
                            "function": "regex",
                            "regex_pattern": r"#### (\d+)",
                            "group_select": index,
                        },
                        {"function": "take_first"},
                    ],
                }
            ],
            "metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
        }
    )
    task.build_all_requests()
    task.instances[0].resps = ["#### 12 #### 34 #### 56"]
    task.apply_filters()
    selected = task.instances[0].filtered_resps["selected"]
    assert selected == expected
    assert task.process_results(task.eval_docs[0], [selected])["exact_match"] == 1


@pytest.mark.parametrize("index", [0, 1, -1, -2])
def test_match_storage_is_bounded_for_long_responses(index):
    response = " ".join(str(100000 + number) for number in range(100000))
    extractor = RegexFilter(r"(\d+)", group_select=index)
    expected = str(100000 + (index if index >= 0 else 100000 + index))
    gc.collect()
    tracemalloc.start()
    try:
        result = extractor.apply([[response]], [{}])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == [[expected]]
    # The response already exists. Selecting the first/last two matches need not
    # allocate the roughly 6 MiB list of all 100,000 captured strings.
    assert peak < 256 * 1024
