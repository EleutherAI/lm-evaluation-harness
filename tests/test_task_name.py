from lm_eval.api.task import Task
from lm_eval.config.task import TaskConfig


class DirectTask(Task):
    def __init__(self, task_name=None):
        self._config = TaskConfig(task=task_name)

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


class HarnessNamedTask(DirectTask):
    EVAL_HARNESS_NAME = "harness_name"


def test_direct_task_name_falls_back_to_stable_class_name():
    task = DirectTask()

    assert task.task_name == "DirectTask"
    assert task.task_name == task.task_name


def test_direct_task_name_uses_eval_harness_name():
    assert HarnessNamedTask().task_name == "harness_name"


def test_configured_task_name_takes_precedence():
    assert HarnessNamedTask(task_name="configured_name").task_name == "configured_name"
