import random
import sys
from unittest.mock import patch

import pytest
from datasets import Dataset

from lm_eval._cli import HarnessCLI
from lm_eval.api.task import ConfigurableTask
from lm_eval.caching import cache
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.evaluator import evaluate, simple_evaluate
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager


class EchoLM(DummyLM):
    tokenizer = None

    def generate_until(self, requests):
        return [request.args[0] for request in requests]


def numbered_task(name, size):
    docs = Dataset.from_dict({"text": [f"row {i}" for i in range(size)]})
    return ConfigurableTask(
        config={
            "task": name,
            "custom_dataset": lambda: {"test": docs},
            "test_split": "test",
            "doc_to_text": "text",
            "doc_to_target": "text",
            "output_type": "generate_until",
            "metric_list": [{"metric": "exact_match"}],
        }
    )


@pytest.fixture
def task():
    return numbered_task("numbered", 23)


def evaluated_ids(task, **kwargs):
    result = evaluate(
        lm=EchoLM(),
        task_dict={"tasks": {"numbered": task}},
        bootstrap_iters=0,
        **kwargs,
    )
    rows = result["samples"]["numbered"]
    for row in rows:
        expected = f"row {row['doc_id']}"
        assert row["doc"]["text"] == row["target"] == row["arguments"][0][0] == expected
        assert row["filtered_resps"] == [expected]
        assert row["exact_match"] == 1
    assert result["n-samples"]["numbered"]["effective"] == len(rows)
    return [row["doc_id"] for row in rows]


def test_seeded_limit(task):
    assert evaluated_ids(task, limit=5) == list(range(5))
    state = random.getstate()
    first = evaluated_ids(task, limit=0.2, sample_seed=0)
    assert random.getstate() == state
    assert len(first) == len(set(first)) == 5  # ceil(23 * 0.2)
    assert first != list(range(5))
    assert evaluated_ids(task, limit=5, sample_seed=0) == first
    assert evaluated_ids(task, limit=5, sample_seed=42) != first
    assert evaluated_ids(task, limit=50, sample_seed=0) == list(range(23))


def test_fraction_is_per_task_and_independent_of_task_order(task):
    tasks = {"numbered": task, "short": numbered_task("short", 8)}

    def run(task_map):
        result = evaluate(
            lm=EchoLM(),
            task_dict={"tasks": task_map},
            limit=0.2,
            sample_seed=42,
            bootstrap_iters=0,
        )
        return {
            name: [row["doc_id"] for row in rows]
            for name, rows in result["samples"].items()
        }

    ids = run(tasks)
    assert len(ids["numbered"]) == 5
    assert len(ids["short"]) == 2
    assert run(dict(reversed(tasks.items()))) == ids


def test_subset_request_cache(task, tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    with patch.object(
        task, "construct_requests", wraps=task.construct_requests
    ) as build:
        ids = evaluated_ids(task, limit=5, sample_seed=0, cache_requests=True)
        assert build.call_count == 5
        build.reset_mock()
        # An explicit selection in a different order hits the same cache entry.
        assert (
            evaluated_ids(task, samples={"numbered": ids[::-1]}, cache_requests=True)
            == ids
        )
        build.assert_not_called()
        other = evaluated_ids(task, limit=5, sample_seed=42, cache_requests=True)
        assert other != ids
        assert build.call_count == 5
        fewer = evaluated_ids(task, limit=3, sample_seed=42, cache_requests=True)
        assert len(fewer) == 3
        assert build.call_count == 8
        assert evaluated_ids(task, cache_requests=True) == list(range(23))
        build.reset_mock()
        assert evaluated_ids(task, limit=4, cache_requests=True) == list(range(4))
        assert evaluated_ids(task, limit=5, sample_seed=0, cache_requests=True) == ids
        build.assert_not_called()


@pytest.mark.parametrize("cache_requests", [False, True])
def test_sharding_preserves_doc_ids(task, tmp_path, monkeypatch, cache_requests):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))
    selected = evaluated_ids(task, limit=5, sample_seed=42)
    for options in ({"samples": selected[::-1]}, {"limit": 5}):
        expected = selected if "samples" in options else list(range(5))
        for rank in range(2):
            for _ in range(2):  # also exercise cache hits
                task.build_all_requests(
                    **options, rank=rank, world_size=2, cache_requests=cache_requests
                )
                assert [req.doc_id for req in task.instances] == expected[rank::2]
                assert list(task.doc_iterator(**options, rank=rank, world_size=2)) == [
                    (idx, {"text": f"row {idx}"}) for idx in expected[rank::2]
                ]


def test_simple_evaluate_and_cli_config(task, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["lm-eval", "run", "--limit", "4", "--sample_seed", "0"]
    )
    cfg = EvaluatorConfig.from_cli(HarnessCLI().parse_args())
    assert cfg.sample_seed == 0
    result = simple_evaluate(
        model=EchoLM(),
        tasks=[task],
        task_manager=TaskManager(include_defaults=False),
        limit=cfg.limit,
        sample_seed=cfg.sample_seed,
        bootstrap_iters=0,
    )
    assert result["config"]["sample_seed"] == 0
    assert [row["doc_id"] for row in result["samples"]["numbered"]] == evaluated_ids(
        task, limit=4, sample_seed=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_seed": 0},
        {"sample_seed": 42, "limit": 0},
        {"samples": {"numbered": [2]}, "limit": 2, "sample_seed": 42},
    ],
)
def test_incompatible_sampling_arguments(task, kwargs):
    with pytest.raises(ValueError, match="--limit"):
        evaluate(lm=EchoLM(), task_dict={"tasks": {"numbered": task}}, **kwargs)


def test_seeded_limit_with_more_ranks_than_documents(task):
    lm = EchoLM()
    lm._world_size = 2
    with pytest.raises(ValueError, match="sample size 1 is too small for 2 ranks"):
        evaluate(
            lm=lm, task_dict={"tasks": {"numbered": task}}, limit=1, sample_seed=42
        )
