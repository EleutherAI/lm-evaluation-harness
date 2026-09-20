import subprocess
import sys

import pytest
from datasets import Dataset, DatasetDict

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.filters import build_filter_ensemble
from lm_eval.filters.extraction import RegexFilter


@pytest.mark.parametrize(
    "pattern",
    [
        r"(?s)\s+return 42\s+",
        r"(?s)(\s+return 42\s+)",
        r"missing|(\s+return 42\s+)|other(.*)",
    ],
)
@pytest.mark.parametrize("kwargs", [{}, {"strip": True}, {"strip": False}])
def test_regex_filter_whitespace(pattern, kwargs):
    response = "\n    return 42\t\n"
    filt = RegexFilter(regex_pattern=pattern, **kwargs)

    expected = response if kwargs.get("strip") is False else "return 42"
    assert filt.apply([[response, response], [response]], [{}, {}]) == [
        [expected, expected],
        [expected],
    ]


def test_regex_filter_preserves_selected_match():
    filt = RegexFilter(
        regex_pattern=r"<code>(.*?)</code>", group_select=-1, strip=False
    )

    assert filt.apply([["<code> first </code><code>\tlast </code>"]], [{}]) == [
        ["\tlast "]
    ]


@pytest.mark.parametrize("response", ["no match", None])
def test_regex_filter_preserves_unmatched_fallback(response):
    filt = RegexFilter(
        regex_pattern=r"<code>(.*?)</code>", fallback=" missing ", strip=False
    )

    assert filt.apply([[response]], [{}]) == [[" missing "]]


def test_regex_filter_preserves_empty_capture_fallback():
    filt = RegexFilter(regex_pattern=r"()()", fallback=" missing ", strip=False)

    assert filt.apply([["text"]], [{}]) == [[" missing "]]


def test_regex_filter_preserves_whitespace_only_match():
    filt = RegexFilter(regex_pattern=r"( +)|(\t+)", strip=False)

    assert filt.apply([["   ", "\t"]], [{}]) == [["   ", "\t"]]


@pytest.mark.parametrize("remove_whitespace", [False, True])
def test_regex_filter_whitespace_option_in_pipeline(remove_whitespace):
    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=(),
        idx=0,
        resps=["<code>    return 42\n</code>"],
    )
    components = [
        ("regex", {"regex_pattern": r"(?s)<code>(.*?)</code>", "strip": False})
    ]
    if remove_whitespace:
        components.append(("remove_whitespace", None))
    components.append(("take_first", None))
    build_filter_ensemble("code", components).apply([instance])

    expected = "return 42" if remove_whitespace else "    return 42\n"
    assert instance.filtered_resps["code"] == expected


def test_regex_filter_preserves_code_completion_scores():
    # Fixed, trusted completions: two correct programs and a wrong-answer control.
    # Exercise request generation, configured filters, execution and aggregation
    # without downloading a model or a benchmark dataset.
    docs = [
        {
            "prompt": "def answer():\n",
            "completion": "    return 42\n",
            "test": "assert answer() == 42",
        },
        {
            "prompt": "def twice(x):\n",
            "completion": "    result = x * 2\n    return result\n",
            "test": "assert twice(3) == 6",
        },
        {
            "prompt": "def add(a, b):\n",
            "completion": "    return a - b\n",
            "test": "assert add(2, 3) == 5",
        },
    ]

    class FixedCompletions(LM):
        def generate_until(self, requests):
            return [request.doc["completion"] for request in requests]

        def loglikelihood(self, requests):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

    def score(doc, results):
        program = doc["prompt"] + results[0] + "\n" + doc["test"]
        result = subprocess.run(  # noqa: S603 - executes only the literal fixtures above
            [sys.executable, "-c", program], capture_output=True, timeout=5, check=False
        )
        return {"pass@1": float(result.returncode == 0)}

    task = ConfigurableTask(
        config={
            "task": "regex_code_fixture",
            "custom_dataset": lambda **kwargs: DatasetDict(
                {"test": Dataset.from_list(docs)}
            ),
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "prompt",
            "doc_to_target": "test",
            "num_fewshot": 0,
            "generation_kwargs": {"until": [], "do_sample": False},
            "process_results": score,
            "metric_list": [
                {"metric": "pass@1", "aggregation": "mean", "higher_is_better": True}
            ],
            "filter_list": [
                {"name": "unfiltered", "filter": [{"function": "take_first"}]},
                {
                    "name": "default",
                    "filter": [
                        {"function": "regex", "regex_pattern": r"(?s)(.+)"},
                        {"function": "take_first"},
                    ],
                },
                {
                    "name": "preserved",
                    "filter": [
                        {
                            "function": "regex",
                            "regex_pattern": r"(?s)(.+)",
                            "strip": False,
                        },
                        {"function": "take_first"},
                    ],
                },
            ],
        }
    )
    task.set_fewshot_seed(0)
    result = evaluate(
        lm=FixedCompletions(), task_dict={"regex_code_fixture": task}, bootstrap_iters=0
    )
    scores = result["results"]["regex_code_fixture"]

    assert scores["pass@1,unfiltered"] == 2 / 3
    assert scores["pass@1,default"] == 0
    assert scores["pass@1,preserved"] == 2 / 3
