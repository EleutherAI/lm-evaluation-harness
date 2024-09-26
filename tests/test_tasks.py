import os
from itertools import islice

import pytest

import lm_eval.tasks as tasks
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator_utils import get_task_list

from .utils import new_tasks


os.environ["TOKENIZERS_PARALLELISM"] = "false"
task_manager = tasks.TaskManager()
# Default Task
TASKS = ["arc_easy"]


def task_class():
    global TASKS
    # CI: new_tasks checks if any modifications have been made
    task_classes = new_tasks()
    # Check if task_classes is empty
    task_classes = task_classes if task_classes else TASKS
    res = tasks.get_task_dict(task_classes, task_manager)
    res = [x.task for x in get_task_list(res)]

    return res


@pytest.fixture()
def limit() -> int:
    return 10


# Tests
@pytest.mark.parametrize("task_class", task_class(), ids=lambda x: f"{x.config.task}")
class TestNewTasks:
    def test_download(self, task_class: ConfigurableTask):
        task_class.download()
        assert task_class.dataset is not None

    def test_has_training_docs(self, task_class: ConfigurableTask):
        assert task_class.has_training_docs() in [True, False]

    def test_check_training_docs(self, task_class: ConfigurableTask):
        if task_class.has_training_docs():
            assert task_class._config["training_split"] is not None

    def test_has_validation_docs(self, task_class):
        assert task_class.has_validation_docs() in [True, False]

    def test_check_validation_docs(self, task_class):
        if task_class.has_validation_docs():
            assert task_class._config["validation_split"] is not None

    def test_has_test_docs(self, task_class):
        assert task_class.has_test_docs() in [True, False]

    def test_check_test_docs(self, task_class):
        task = task_class
        if task.has_test_docs():
            assert task._config["test_split"] is not None

    def test_should_decontaminate(self, task_class):
        task = task_class
        assert task.should_decontaminate() in [True, False]
        if task.should_decontaminate():
            assert task._config["doc_to_decontamination_query"] is not None

    def test_doc_to_text(self, task_class, limit):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        _array = [task.doc_to_text(doc) for doc in arr]
        # space convention; allow txt to have length 0 for perplexity-like tasks since the model tacks an <|endoftext|> on
        assert all(
            isinstance(x, str) and (x[-1] != " " if len(x) != 0 else True)
            for x in _array
        )

    def test_create_choices(self, task_class, limit):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        if "multiple_choice" in task._config.output_type:
            _array = [task.doc_to_choice(doc) for doc in arr]
            assert all(isinstance(x, list) for x in _array)
            assert all(isinstance(x[0], str) for x in _array)

    def test_doc_to_target(self, task_class, limit):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        _array_target = [task.doc_to_target(doc) for doc in arr]
        if task._config.output_type == "multiple_choice":
            # TODO<baber>: label can be string or int; add better test conditions
            assert all(
                (isinstance(label, int) or isinstance(label, str))
                for label in _array_target
            )

    def test_build_all_requests(self, task_class, limit):
        task_class.build_all_requests(rank=1, limit=limit, world_size=1)
        assert task_class.instances is not None

    # ToDO: Add proper testing
    def test_construct_requests(self, task_class, limit):
        task = task_class
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        requests = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
        assert len(requests) == limit if limit else True
