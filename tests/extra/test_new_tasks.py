import pytest
from itertools import islice
import lm_eval.tasks as tasks
from .utilities_testing import load_changed_files, parser
from typing import List, ClassVar
from lm_eval.api.task import ConfigurableTask
import os


# GitHub CI
# If tasks folder has changed then we get the list of files from FILENAME
# and parse the yaml files to get the task names.
# Or if API has changed then we set the ENV variable API to True
# and run some given extended tasks
def new_tasks() -> List[str]:
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if os.path.exists(FILENAME):
        return parser(load_changed_files(FILENAME))
    elif os.getenv("API") is not None:
        return ["arc_easy", "hellaswag", "piqa", "wikitext"]
    # if both not true just do arc_easy
    else:
        return ["arc_easy"]


@pytest.fixture(params=new_tasks())
def task_class(request) -> ConfigurableTask:
    task_name = request.param
    if task_name is None:
        task_name = "arc_easy"
    x = [cls for name, cls in tasks.TASK_REGISTRY.items() if name == task_name]
    return x[0]


@pytest.fixture(params=new_tasks())
def limit(request) -> int:
    # not used; just for consistency
    return 100


# Tests
def test_download(task_class: ConfigurableTask):
    task_class().download()
    assert task_class().dataset is not None


def test_has_training_docs(task_class: ConfigurableTask):
    assert task_class().has_training_docs() in [True, False]


def test_check_training_docs(task_class: ConfigurableTask):
    task = task_class()
    assert task.has_training_docs() if task._config["training_split"] else True


def test_has_validation_docs(task_class):
    assert task_class().has_training_docs() in [True, False]


def test_check_validation_docs(task_class):
    task = task_class()
    assert (
        task_class().has_training_docs() if task._config["validation_split"] else True
    )


def test_has_test_docs(task_class):
    assert task_class().has_training_docs() in [True, False]


def test_check_test_docs(task_class):
    task = task_class()
    assert task_class().has_training_docs() if task._config["test_split"] else True


def test_should_decontaminate(task_class):
    task_class = task_class()
    assert task_class.should_decontaminate() in [True, False]
    if task_class.should_decontaminate():
        assert task_class._config["doc_to_decontamination_query"] is not None


def test_doc_to_text(task_class, limit):
    arr = (
        list(islice(task_class().test_docs(), limit))
        if limit
        else list(task_class().test_docs())
    )
    _array = [task_class().doc_to_text(doc) for doc in arr]
    # space convention; allow txt to have length 0 for perplexity-like tasks since the model tacks an <|endoftext|> on
    assert all(
        isinstance(x, str) and (x[-1] != " " if len(x) != 0 else True) for x in _array
    )


def test_create_choices(task_class, limit):
    arr = (
        list(islice(task_class().test_docs(), limit))
        if limit
        else list(task_class().test_docs())
    )
    _array = [task_class().doc_to_choice(doc) for doc in arr]
    # assert all(len(x) == 4 for x in _array)
    assert all(isinstance(x, list) for x in _array)
    assert all(isinstance(x[0], str) for x in _array)


def test_doc_to_target(task_class, limit):
    arr = (
        list(islice(task_class().test_docs(), limit))
        if limit
        else list(task_class().test_target())
    )
    _array_target = [task_class().doc_to_target(doc) for doc in arr]
    assert all(isinstance(label, int) for label in _array_target)
    assert len(_array_target) == limit if limit else True
    # _array_text = [task.doc_to_text(doc) for doc in arr]
    # Not working
    # assert all(tgt[0] == " " or txt[-1] == "\n" if  len(txt) != 0 else True for txt, tgt in zip(_array_text, _array_target))


def test_build_all_requests(task_class, limit):
    task_class().build_all_requests(rank=1, limit=limit, world_size=1)
    assert task_class.instances is not None


def test_construct_requests(task_class, limit):
    arr = (
        list(islice(task_class().test_docs(), limit))
        if limit
        else list(task_class().test_docs())
    )
    requests = [
        task_class().construct_requests(doc, task_class().doc_to_text(doc))
        for doc in arr
    ]
    assert all(isinstance(doc, list) for doc in requests)
    assert len(requests) == limit if limit else True
