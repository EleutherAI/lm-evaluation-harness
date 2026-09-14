import pytest

from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.models.dummy import DummyLM


class _SampleLoggingTask(Task):
    OUTPUT_TYPE = "generate_until"

    def download(self, *args, **kwargs) -> None:
        self.dataset = {
            "test": [
                {"marker": index, "prompt": f"document {index}"} for index in range(10)
            ]
        }

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc) -> str:
        return doc["prompt"]

    def doc_to_target(self, doc) -> str:
        return "lol"

    def construct_requests(self, doc, ctx, metadata=None, **kwargs):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
            metadata=metadata,
        )

    def process_results(self, doc, results):
        return {"score": 1.0}

    def aggregation(self):
        return {"score": mean}

    def higher_is_better(self):
        return {"score": True}


def _evaluate_sample_logging(samples: list[int]):
    task = _SampleLoggingTask(config={"task": "sample_logging", "num_fewshot": 0})
    task.set_fewshot_seed(0)
    result = evaluator.evaluate(
        lm=DummyLM(),
        task_dict={"tasks": {"sample_logging": task}, "groups": {}},
        samples={"sample_logging": samples},
        bootstrap_iters=0,
        log_samples=True,
    )
    assert result is not None
    return task, result["samples"]["sample_logging"]


@pytest.mark.parametrize("samples", [[5, 2, 9], [5, 2, 2, 9]])
def test_unsorted_samples_log_original_document_ids(samples):
    task, logged_samples = _evaluate_sample_logging(samples)

    assert [
        (sample["doc_id"], sample["doc"]["marker"]) for sample in logged_samples
    ] == [(2, 2), (5, 5), (9, 9)]
    assert [
        (instance.doc_id, instance.doc["marker"]) for instance in task.instances
    ] == [(0, 2), (1, 5), (2, 9)]


def test_empty_samples_preserve_unfiltered_logging():
    task, logged_samples = _evaluate_sample_logging([])

    assert [
        (sample["doc_id"], sample["doc"]["marker"]) for sample in logged_samples
    ] == [(index, index) for index in range(10)]
    assert [instance.doc_id for instance in task.instances] == list(range(10))


def test_sample_iterator_preserves_global_ordinals_across_ranks():
    task = _SampleLoggingTask(config={"task": "sample_logging", "num_fewshot": 0})
    samples = [5, 2, 9]

    rank_zero = list(task.doc_iterator(rank=0, world_size=2, samples=samples))
    rank_one = list(task.doc_iterator(rank=1, world_size=2, samples=samples))

    assert [(doc_id, doc["marker"]) for doc_id, doc in rank_zero] == [
        (0, 2),
        (2, 9),
    ]
    assert [(doc_id, doc["marker"]) for doc_id, doc in rank_one] == [(1, 5)]


def test_out_of_range_sample_still_raises():
    task = _SampleLoggingTask(config={"task": "sample_logging", "num_fewshot": 0})

    with pytest.raises(AssertionError, match="interval"):
        list(task.doc_iterator(samples=[10]))
