from itertools import islice

import pytest


pytest.skip(
    "Need to confirm if module tests are still compatible", allow_module_level=True
)


from lm_eval import tasks as tasks
from lm_eval.api.task import ConfigurableTask
from tests.test_tasks import BaseTasks, task_class


@pytest.fixture()
def limit() -> int:
    return 10


@pytest.mark.parametrize(
    "task_class",
    task_class(
        ["arc_easy_unitxt"], tasks.TaskManager(include_path="./tests/testconfigs")
    ),
    ids=lambda x: f"{x.config.task}",
)
class TestUnitxtTasks(BaseTasks):
    """
    Test class for Unitxt tasks parameterized with a small custom
    task as described here:
      https://www.unitxt.ai/en/latest/docs/lm_eval.html
    """

    def test_check_training_docs(self, task_class: ConfigurableTask):
        if task_class.has_training_docs():
            assert task_class.dataset["train"] is not None

    def test_check_validation_docs(self, task_class):
        if task_class.has_validation_docs():
            assert task_class.dataset["validation"] is not None

    def test_check_test_docs(self, task_class):
        task = task_class
        if task.has_test_docs():
            assert task.dataset["test"] is not None

    def test_doc_to_text(self, task_class, limit: int):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        _array = [task.doc_to_text(doc) for doc in arr]
        if not task.multiple_input:
            for x in _array:
                assert isinstance(x, str)
        else:
            pass
