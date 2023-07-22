import pytest
from itertools import islice
import lm_eval.tasks as tasks
from .utilities_testing import load_changed_files, parser
from typing import List
from lm_eval.api.task import ConfigurableTask
import os


# GitHub CI
def new_tasks() -> List[str]:
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if os.path.exists(FILENAME):
        # If tasks folder has changed then we get the list of files from FILENAME
        # and parse the yaml files to get the task names.
        return parser(load_changed_files(FILENAME))
    elif os.getenv("API") is not None:
        # Or if API has changed then we set the ENV variable API to True
        # and run  given tasks.
        return ["arc_easy", "hellaswag", "piqa", "wikitext"]
    # if both not true just do arc_easy
    else:
        return ["arc_easy"]


def get_task_class() -> List[ConfigurableTask]:
    task_name = new_tasks()
    x = [cls for name, cls in tasks.TASK_REGISTRY.items() if name in task_name]
    return x


@pytest.fixture()
def limit() -> int:
    return 10


# Tests
@pytest.mark.parametrize("task_class", get_task_class())
class TestNewTasks:
    def test_download(self, task_class: ConfigurableTask):
        task_class().download()
        assert task_class().dataset is not None

    def test_has_training_docs(self, task_class: ConfigurableTask):
        assert task_class().has_training_docs() in [True, False]

    def test_check_training_docs(self, task_class: ConfigurableTask):
        task = task_class()
        if task.has_training_docs():
            assert task._config["training_split"] is not None

    def test_has_validation_docs(self, task_class):
        assert task_class().has_validation_docs() in [True, False]

    def test_check_validation_docs(self, task_class):
        task = task_class()
        if task.has_validation_docs():
            assert task._config["validation_split"] is not None

    def test_has_test_docs(self, task_class):
        assert task_class().has_test_docs() in [True, False]

    def test_check_test_docs(self, task_class):
        task = task_class()
        if task.has_test_docs():
            assert task._config["test_split"] is not None

    def test_should_decontaminate(self, task_class):
        task = task_class()
        assert task.should_decontaminate() in [True, False]
        if task.should_decontaminate():
            assert task._config["doc_to_decontamination_query"] is not None

    def test_doc_to_text(self, task_class, limit):
        task = task_class()
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
        task = task_class()
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        if "multiple_choice" in task._config.group:
            _array = [task.doc_to_choice(doc) for doc in arr]
            # assert all(len(x) == 4 for x in _array)
            assert all(isinstance(x, list) for x in _array)
            assert all(isinstance(x[0], str) for x in _array)

    def test_doc_to_target(self, task_class, limit):
        task = task_class()
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        _array_target = [task.doc_to_target(doc) for doc in arr]
        assert all(isinstance(label, int) for label in _array_target)
        assert len(_array_target) == limit if limit else True
        # _array_text = [task.doc_to_text(doc) for doc in arr]
        # Not working
        # assert all(tgt[0] == " " or txt[-1] == "\n" if  len(txt) != 0 else True for txt, tgt in zip(_array_text, _array_target))

    def test_build_all_requests(self, task_class, limit):
        task_class().build_all_requests(rank=1, limit=limit, world_size=1)
        assert task_class.instances is not None

    def test_construct_requests(self, task_class, limit):
        task = task_class()
        arr = (
            list(islice(task.test_docs(), limit))
            if task.has_test_docs()
            else list(islice(task.validation_docs(), limit))
        )
        requests = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
        assert all(isinstance(doc, list) for doc in requests)
        assert len(requests) == limit if limit else True
