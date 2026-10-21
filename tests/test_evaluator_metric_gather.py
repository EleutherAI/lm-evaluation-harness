"""Compare serial evaluation with simulated transport of real per-rank metrics."""

import pytest
from datasets import Dataset

from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.models.dummy import DummyLM


torch = pytest.importorskip("torch")


class InMemoryTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {"test": Dataset.from_dict({"text": ["0", "1", "2", "3"]})}


class GatheringLM(DummyLM):
    def __init__(self, rank=0, world_size=1, peer=None):
        super().__init__()
        self._rank, self._world_size, self._device = rank, world_size, "cpu"
        self.peer = iter(peer or [])
        self.sent = []

    def all_gather(self, tensor):
        return torch.stack([tensor, tensor])

    def gather_object(self, obj, dst=0):
        self.sent.append(obj)
        return [obj, next(self.peer)] if self.rank == 0 else None

    def generate_until(self, requests):
        return ["ok"] * len(requests)


def make_task(mode, filter_names):
    def process_results(doc, results):
        value = int(doc["text"])
        metric = "odd" if value % 2 else "even"
        if mode == "disjoint":
            return {metric: value}
        if mode == "worker_empty":
            return {} if value % 2 else {"even": value}
        if mode == "primary_empty":
            return {"odd": value} if value % 2 else {}
        output = {"acc": value}
        if (mode == "worker_extra" and value % 2) or (
            mode == "primary_extra" and not value % 2
        ):
            output[metric] = value
        return output

    return InMemoryTask(
        config={
            "task": "toy",
            "test_split": "test",
            "doc_to_text": "text",
            "doc_to_target": "text",
            "num_fewshot": 0,
            "output_type": "generate_until",
            "process_results": process_results,
            "metric_list": [
                {"metric": name, "aggregation": "mean", "higher_is_better": True}
                for name in ["acc", "even", "odd"]
            ],
            "filter_list": [
                {"name": name, "filter": [{"function": "take_first"}]}
                for name in filter_names
            ],
        }
    )


@pytest.mark.parametrize(
    "mode",
    [
        "common",
        "disjoint",
        "worker_extra",
        "primary_extra",
        "worker_empty",
        "primary_empty",
    ],
)
@pytest.mark.parametrize("filter_names", [["none"], ["none", "other"]])
def test_distributed_metric_union_matches_serial(mode, filter_names):
    def run(lm):
        return evaluate(
            lm,
            {"tasks": {"toy": make_task(mode, filter_names)}},
            bootstrap_iters=0,
            log_samples=True,
        )

    serial = run(GatheringLM())
    worker = GatheringLM(rank=1, world_size=2)
    assert run(worker) is None
    parallel = run(GatheringLM(rank=0, world_size=2, peer=worker.sent))

    assert parallel["results"] == serial["results"]
    assert parallel["n-samples"] == serial["n-samples"]
    by_document = lambda sample: (sample["doc_id"], sample["filter"])
    assert sorted(parallel["samples"]["toy"], key=by_document) == sorted(
        serial["samples"]["toy"], key=by_document
    )
