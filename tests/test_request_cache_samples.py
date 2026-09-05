"""Request caching must preserve the selected documents, not just the task name."""

from unittest.mock import Mock

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.caching import cache
from lm_eval.evaluator import evaluate
from lm_eval.models.dummy import DummyLM


class SampleTask(ConfigurableTask):
    def __init__(self):
        super().__init__(
            config={
                "task": "cache_samples",
                "dataset_path": "unused",
                "test_split": "test",
                "output_type": "multiple_choice",
                "doc_to_text": "question",
                "doc_to_target": "answer",
                "doc_to_choice": ["0", "1"],
                "num_fewshot": 0,
                "metric_list": [
                    {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
                ],
            }
        )

    def download(self, *args, **kwargs):
        self.dataset = {
            "test": datasets.Dataset.from_list(
                [{"question": str(i), "answer": i % 2} for i in range(6)]
            )
        }


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))


def request_contents(task):
    return [(req.doc_id, req.doc, req.args) for req in task.instances]


@pytest.mark.parametrize("rank,world_size", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize(
    "first_samples,second_samples",
    [(None, [1, 3]), ([0, 2], [1, 3]), ([1, 3], None)],
)
def test_changed_selection_matches_uncached_requests(
    first_samples, second_samples, rank, world_size
):
    first = SampleTask()
    first.build_all_requests(
        samples=first_samples, rank=rank, world_size=world_size, cache_requests=True
    )

    second = SampleTask()
    second.build_all_requests(
        samples=second_samples, rank=rank, world_size=world_size, cache_requests=True
    )
    uncached = SampleTask()
    uncached.build_all_requests(
        samples=second_samples, rank=rank, world_size=world_size
    )

    assert request_contents(second) == request_contents(uncached)


@pytest.mark.parametrize(
    "first_samples,second_samples", [([1, 3], [3, 1, 1]), (None, [])]
)
def test_equivalent_selections_reuse_cache(first_samples, second_samples):
    first = SampleTask()
    first.build_all_requests(samples=first_samples, cache_requests=True)
    second = SampleTask()
    second.construct_requests = Mock(side_effect=AssertionError("cache miss"))
    second.build_all_requests(samples=second_samples, cache_requests=True)

    assert request_contents(second) == request_contents(first)
    second.construct_requests.assert_not_called()


def test_cached_requests_do_not_bypass_sample_validation():
    first = SampleTask()
    first.build_all_requests(cache_requests=True)

    with pytest.raises(AssertionError, match="Elements of --samples"):
        SampleTask().build_all_requests(samples=[0, 6], cache_requests=True)


def test_limited_run_still_caches_all_documents():
    first = SampleTask()
    first.build_all_requests(limit=1, cache_requests=True)
    assert len(first.instances) == 2

    full = SampleTask()
    full.construct_requests = Mock(side_effect=AssertionError("cache miss"))
    full.build_all_requests(cache_requests=True)

    assert [req.doc["question"] for req in full.instances[::2]] == [
        str(i) for i in range(6)
    ]
    full.construct_requests.assert_not_called()


def test_legacy_cache_cannot_supply_an_unknown_selection():
    first = SampleTask()
    first.build_all_requests(samples=[0, 2])
    legacy_key = "requests-cache_samples-0shot-rank0-world_size1-tokenizer"
    cache.save_to_cache(legacy_key, [first.instances[:2], first.instances[2:]])

    full = SampleTask()
    full.build_all_requests(cache_requests=True)
    assert [req.doc["question"] for req in full.instances[::2]] == [
        str(i) for i in range(6)
    ]


def test_cached_selection_preserves_evaluation_score():
    class ParityLM(DummyLM):
        def loglikelihood(self, requests, **kwargs):
            return [
                (0.0 if int(req.args[0]) % 2 == int(req.args[1]) else -1.0, False)
                for req in requests
            ]

    first = SampleTask()
    first.build_all_requests(samples=[0, 2], cache_requests=True)

    results = evaluate(
        lm=ParityLM(),
        task_dict={"cache_samples": SampleTask()},
        samples={"cache_samples": [1, 3]},
        cache_requests=True,
        bootstrap_iters=0,
        log_samples=False,
    )

    assert results["results"]["cache_samples"]["acc,none"] == 1.0
