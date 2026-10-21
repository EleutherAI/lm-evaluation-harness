"""Exercise distributed dispatch with in-memory tasks and simulated collectives."""

import pytest
from datasets import Dataset

from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.dummy import DummyLM


torch = pytest.importorskip("torch")


class InMemoryTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {"test": Dataset.from_dict({"text": ["0", "1", "2"]})}


class CountingLM(DummyLM):
    def __init__(self, rank, world_size, peer_metrics=None):
        super().__init__()
        self._rank, self._world_size, self._device = rank, world_size, "cpu"
        self.peer_metrics = peer_metrics
        self.requests = []

    def all_gather(self, tensor):
        # Three documents shard into two on rank 0 and one on rank 1.
        # Infer the expansion factor from the value actually being gathered.
        local_documents = 2 if self.rank == 0 else 1
        repeats = tensor.item() // local_documents
        return torch.tensor([2 * repeats, repeats])

    def gather_object(self, obj, dst=0):
        self.metrics = obj
        return [obj, self.peer_metrics] if self.rank == 0 else None

    def generate_until(self, requests):
        self.requests = requests
        return ["ok"] * len(requests)


def make_tasks(repeats):
    return {
        f"task_{i}": InMemoryTask(
            config={
                "task": f"task_{i}",
                "test_split": "test",
                "doc_to_text": "text",
                "doc_to_target": "text",
                "num_fewshot": 0,
                "output_type": "generate_until",
                "repeats": count,
                "process_results": lambda doc, results: {"acc": 1.0},
                "metric_list": [
                    {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
                ],
            }
        )
        for i, count in enumerate(repeats)
    }


@pytest.mark.parametrize("repeats", [(1,), (1, 1), (2, 2), (2, 1), (1, 2), (2, 3, 1)])
def test_distributed_padding_counts_each_tasks_repeats(repeats):
    worker = CountingLM(rank=1, world_size=2)
    evaluate(
        worker, {"tasks": make_tasks(repeats)}, log_samples=False, bootstrap_iters=0
    )
    primary = CountingLM(rank=0, world_size=2, peer_metrics=worker.metrics)
    result = evaluate(
        primary, {"tasks": make_tasks(repeats)}, log_samples=False, bootstrap_iters=0
    )

    expected = 2 * sum(repeats)
    assert len(primary.requests) == len(worker.requests) == expected
    assert all(values["effective"] == 3 for values in result["n-samples"].values())


def test_single_process_does_not_pad_mixed_repeats():
    lm = CountingLM(rank=0, world_size=1)
    evaluate(lm, {"tasks": make_tasks([2, 1])}, log_samples=False, bootstrap_iters=0)
    assert len(lm.requests) == 9
