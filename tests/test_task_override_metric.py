from lm_eval.api.task import Task
from lm_eval.config.task import TaskConfig


class DirectTask(Task):
    def __init__(self):
        self._config = TaskConfig()

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

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


def test_override_metric_invokes_bypass_for_direct_task():
    task = DirectTask()
    results = ["generated output"]

    task.override_metric("bypass")

    assert task.process_results({}, results) == {"bypass": None}
    assert not callable(task.process_results({}, results)["bypass"])
