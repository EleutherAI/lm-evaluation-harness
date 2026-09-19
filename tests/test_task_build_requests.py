import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.task import Task
from lm_eval.config.task import TaskConfig


class TinyGenerateTask(Task):
    VERSION = 0
    OUTPUT_TYPE = "generate_until"

    def __init__(self, docs):
        self._docs = docs
        self._config = TaskConfig(
            task="tiny_generate",
            num_fewshot=0,
            output_type="generate_until",
            repeats=1,
        )
        self._instances = None
        self._filters = []
        self._training_docs = None
        self._fewshot_docs = None
        self.fewshot_rnd = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self._docs

    def validation_docs(self):
        return []

    def doc_to_text(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return doc["target"]

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {}),
            idx=0,
            metadata=kwargs["metadata"],
        )

    def process_results(self, doc, results):
        return {}

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}


def test_build_all_requests_allows_empty_nonzero_distributed_rank():
    task = TinyGenerateTask([{"text": "question", "target": "answer"}])

    task.build_all_requests(rank=1, world_size=2)

    assert task.instances == []


def test_build_all_requests_still_rejects_empty_rank_zero_task():
    task = TinyGenerateTask([])

    with pytest.raises(ValueError, match="did not find any docs"):
        task.build_all_requests(rank=0, world_size=2)
