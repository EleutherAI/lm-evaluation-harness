import json
import math
from collections import Counter
from fractions import Fraction

import datasets
import numpy as np
import pytest

from lm_eval.api.group import Group
from lm_eval.api.metrics import bayes_at_n_fn, bayes_at_n_mean, bayes_at_n_std
from lm_eval.api.model import LM
from lm_eval.api.registry import get_aggregation, get_metric, get_metric_aggregation
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.group import AggMetricConfig
from lm_eval.evaluator import evaluate
from lm_eval.utils import make_table


@pytest.mark.parametrize(
    "rows, expected_mean, expected_variance",
    [
        ([[0, 0], [1, 1]], Fraction(1, 2), Fraction(3, 160)),
        ([[0, 1], [0, 1]], Fraction(1, 2), Fraction(1, 40)),
        ([[0]], Fraction(1, 3), Fraction(1, 18)),
        ([[1]], Fraction(2, 3), Fraction(1, 18)),
        ([[1, 1, 1], [0, 1, 0], [0, 0, 0]], Fraction(7, 15), Fraction(7, 675)),
    ],
)
def test_closed_form_cases(rows, expected_mean, expected_variance):
    assert bayes_at_n_mean(rows) == pytest.approx(float(expected_mean))
    assert bayes_at_n_std(rows) == pytest.approx(math.sqrt(expected_variance))


@pytest.mark.parametrize("seed", range(5))
def test_representative_workloads_against_fraction_moments(seed):
    rows = np.random.default_rng(seed).integers(0, 2, size=(31, 17)).tolist()
    means, variances = [], []
    for row in rows:
        alpha, beta = 1 + sum(row), 1 + len(row) - sum(row)
        means.append(Fraction(alpha, alpha + beta))
        variances.append(
            Fraction(alpha * beta, (alpha + beta) ** 2 * (alpha + beta + 1))
        )
    assert bayes_at_n_mean(rows) == pytest.approx(
        float(sum(means) / len(rows)), abs=1e-14
    )
    assert bayes_at_n_std(rows) == pytest.approx(
        math.sqrt(sum(variances)) / len(rows), abs=1e-14
    )
    shuffled = [row[::-1] for row in rows[::-1]]
    assert bayes_at_n_mean(shuffled) == pytest.approx(bayes_at_n_mean(rows))
    assert bayes_at_n_std(shuffled) == pytest.approx(bayes_at_n_std(rows))
    shards = [rows[::2], rows[1::2]]
    gathered = [row for shard in shards for row in shard]
    assert bayes_at_n_std(gathered) == pytest.approx(bayes_at_n_std(rows))


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[]],
        [0, 1],
        [[0, 1], [1]],
        [[2]],
        [[0.5]],
        [[float("nan")]],
        [["1"]],
        [[1j]],
    ],
)
@pytest.mark.parametrize("aggregate", [bayes_at_n_mean, bayes_at_n_std])
def test_reject_invalid_outcomes(rows, aggregate):
    with pytest.raises(ValueError, match="rectangular binary"):
        aggregate(rows)


@pytest.mark.parametrize("n", [True, 0, -1, 1.5, "2"])
def test_reject_invalid_trial_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        bayes_at_n_fn(predictions=[["yes", "no"]], references=["yes"], n=n)


@pytest.mark.parametrize(
    "predictions", [["yes"], [["yes"]], [["yes", "no", "no"]], [["yes", None]]]
)
def test_reject_discarded_or_incomplete_responses(predictions):
    with pytest.raises(ValueError, match="exactly 2 responses"):
        bayes_at_n_fn(predictions=predictions, references=["yes"], n=2)


def test_registry_and_exact_match_normalization():
    for name, aggregate in [
        ("bayes_at_n", bayes_at_n_mean),
        ("bayes_at_n_std", bayes_at_n_std),
    ]:
        assert get_metric(name) is bayes_at_n_fn
        assert get_aggregation(name) is aggregate
        assert get_metric_aggregation(name) is aggregate
    assert bayes_at_n_fn(
        predictions=[["Answer: YES!", "Answer: no", "Answer: 1. yes"]],
        references=["yes", "no"],
        n=3,
        regexes_to_ignore=[r"^Answer: "],
        ignore_case=True,
        ignore_punctuation=True,
        ignore_numbers=True,
    ) == (1, 1, 0)


class LocalTask(ConfigurableTask):
    def __init__(
        self, docs, *, name="bayes_local", metrics=None, n=2, filters=True, choices=None
    ):
        self.local_docs = docs
        config = {
            "task": name,
            "dataset_path": "local",
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "num_fewshot": 0,
            "repeats": n,
            "generation_kwargs": {
                "do_sample": True,
                "temperature": 0.8,
                "until": ["\n"],
            },
            "metric_list": metrics
            or [{"metric": "bayes_at_n", "n": n}, {"metric": "bayes_at_n_std", "n": n}],
        }
        if filters:
            config["filter_list"] = [
                {"name": "all", "filter": [{"function": "take_first_k", "k": n}]}
            ]
        if choices:
            config["doc_to_choice"] = choices
        super().__init__(config=config)
        self.set_fewshot_seed(0)

    def download(self, *args, **kwargs):
        self.dataset = datasets.DatasetDict(
            {"test": datasets.Dataset.from_list(self.local_docs)}
        )


class ScriptedLM(LM):
    def __init__(self):
        super().__init__()
        self.calls = Counter()

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        results = []
        for request in requests:
            key = (request.task_name, request.doc_id)
            results.append(request.doc["responses"][self.calls[key]])
            self.calls[key] += 1
        return results


def local_docs():
    return [
        {"question": "first", "answer": "yes", "responses": ["no", "no"]},
        {"question": "second", "answer": "yes", "responses": ["yes", "yes"]},
    ]


def test_public_evaluator_multiple_tasks_and_serialization():
    tasks = {name: LocalTask(local_docs(), name=name) for name in ["one", "two"]}
    model = ScriptedLM()
    result = evaluate(lm=model, task_dict=tasks, bootstrap_iters=10, log_samples=True)
    result = json.loads(json.dumps(result))
    for name in tasks:
        metrics = result["results"][name]
        assert metrics["bayes_at_n,all"] == pytest.approx(0.5)
        assert metrics["bayes_at_n_std,all"] == pytest.approx(
            math.sqrt(Fraction(3, 160))
        )
        assert metrics["bayes_at_n_stderr,all"] == "N/A"
        assert metrics["bayes_at_n_std_stderr,all"] == "N/A"
        assert [sample["bayes_at_n"] for sample in result["samples"][name]] == [
            [0, 0],
            [1, 1],
        ]
    assert set(model.calls.values()) == {2}
    assert "bayes_at_n_std" in make_table(result)


def test_task_adapter_accepted_targets_and_normalization():
    docs = [{"question": "first", "answer": ["yes", "y"], "responses": ["YES!", "y"]}]
    task = LocalTask(
        docs,
        metrics=[
            {
                "metric": "bayes_at_n",
                "n": 2,
                "ignore_case": True,
                "ignore_punctuation": True,
            }
        ],
    )
    result = evaluate(lm=ScriptedLM(), task_dict={"local": task}, bootstrap_iters=0)
    assert result["results"]["local"]["bayes_at_n,all"] == pytest.approx(3 / 4)
    assert result["samples"]["local"][0]["bayes_at_n"] == (1, 1)


def test_task_adapter_choice_target():
    docs = [{"question": "first", "answer": 1, "responses": ["yes", "no"]}]
    task = LocalTask(docs, choices=["no", "yes"])
    assert task.process_results(docs[0], [["yes", "no"]])["bayes_at_n"] == (1, 0)


def test_public_evaluator_rejects_default_filter():
    task = LocalTask(local_docs(), filters=False)
    with pytest.raises(ValueError, match="repeat-preserving filter"):
        evaluate(lm=ScriptedLM(), task_dict={"local": task}, bootstrap_iters=0)


@pytest.mark.parametrize("metric", ["bayes_at_n", "bayes_at_n_std"])
@pytest.mark.parametrize("weight_by_size", [True, False])
def test_public_evaluator_rejects_group_summary(metric, weight_by_size):
    task = LocalTask(local_docs())
    group = Group(
        "bayes_group",
        aggregate_metric_list=[
            AggMetricConfig(metric=metric, weight_by_size=weight_by_size)
        ],
    )
    group.add(task)
    with pytest.raises(ValueError, match="per-task corpus results only"):
        evaluate(
            lm=ScriptedLM(),
            task_dict={"tasks": {task.task_name: task}, "groups": {group.name: group}},
            bootstrap_iters=0,
        )


def test_group_can_aggregate_other_metrics_with_bayes_task_results():
    task = LocalTask(
        local_docs(),
        metrics=[
            {"metric": "bayes_at_n", "n": 2},
            {"metric": "bayes_at_n_std", "n": 2},
            {"metric": "exact_match"},
        ],
    )
    group = Group(
        "other_summary", aggregate_metric_list=[AggMetricConfig(metric="exact_match")]
    )
    group.add(task)
    result = evaluate(
        lm=ScriptedLM(),
        task_dict={"tasks": {task.task_name: task}, "groups": {group.name: group}},
        bootstrap_iters=0,
    )
    assert result["results"][task.task_name]["bayes_at_n,all"] == pytest.approx(0.5)
    assert result["results"][task.task_name]["bayes_at_n_std,all"] == pytest.approx(
        math.sqrt(Fraction(3, 160))
    )
    assert result["groups"][group.name]["exact_match,all"] == pytest.approx(0.5)


@pytest.mark.parametrize("references", [[], [None], [1]])
def test_reject_invalid_targets(references):
    with pytest.raises(ValueError, match="at least one string target"):
        bayes_at_n_fn(predictions=[["yes"]], references=references, n=1)


def test_alternate_targets_preserved_after_another_metric():
    docs = [{"question": "first", "answer": ["yes", "y"], "responses": ["yes", "y"]}]
    task = LocalTask(
        docs,
        metrics=[{"metric": "exact_match"}, {"metric": "bayes_at_n", "n": 2}],
    )
    assert task.process_results(docs[0], [["yes", "y"]])["bayes_at_n"] == (1, 1)
