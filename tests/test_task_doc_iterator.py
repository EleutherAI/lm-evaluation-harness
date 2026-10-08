from lm_eval.api.task import Task
from lm_eval.config.task import TaskConfig


class StubTask(Task):
    def __init__(self):
        self._config = TaskConfig(task="stub_task")
        self._docs = [{"id": 0}, {"id": 1}, {"id": 2}]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self._docs

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return ""

    def construct_requests(self, doc, ctx, **kwargs):
        return []

    def process_results(self, doc, results):
        return {}

    def aggregation(self):
        return {}

    def higher_is_better(self):
        return {}


def test_doc_iterator_with_empty_samples_selects_no_documents():
    task = StubTask()

    assert list(task.doc_iterator(samples=[])) == []


def test_doc_iterator_without_samples_selects_all_documents():
    task = StubTask()

    assert list(task.doc_iterator(samples=None)) == list(enumerate(task._docs))
