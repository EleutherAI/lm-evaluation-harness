import pytest
from itertools import islice
import lm_eval.tasks as tasks
from tests.extra.test_utils import load_changed_files, parser
from typing import List, ClassVar
import os


@pytest.fixture()
def any_new_tasks(request) -> bool:
    return request.config.getoption("--new_task")


# ["arc_easy] else get list of new tasks
def new_tasks(any_new_tasks: bool) -> List[str]:
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if any_new_tasks and os.path.exists(FILENAME):
        return [parser(load_changed_files(FILENAME))]
    elif "API" in os.getenv():
        return ["arc_easy", "hellaswag", "piqa", "wikitext"]
    else:
        return ["arc_easy"]


@pytest.fixture(params=new_tasks(any_new_tasks))
def task_class(request):
    task_name = request.param
    return [cls for name, cls in tasks.TASK_REGISTRY.items() if name in task_name][0]


@pytest.fixture()
def limit(any_new_tasks: bool) -> int:
    return 100 if any_new_tasks else 10


# Tests


def test_download(task_class):
    task_class().download()
    assert task_class().dataset is not None


def test_has_training_docs(task_class):
    assert task_class().has_training_docs() in [True, False]


def test_check_training_docs(task_class):
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


# def test_create_choices(task_class):
#     arr = list(islice(task_class().test_docs(), 1))
#     choices = task_class().create_choices(arr[0])
#     assert choices is not None
# checking if number of choices is correct


# @pytest.mark.parametrize("taskname,task_class", tasks.TASK_REGISTRY.items())
# def test_basic_interface(taskname, task_class):
#     print("Evaluating task", taskname)
#     task = task_class()
#
#     assert task.has_training_docs() in [True, False]
#     assert task.has_validation_docs() in [True, False]
#     assert task.has_test_docs() in [True, False]
#
#     assert isinstance(task.aggregation(), dict)
#     assert isinstance(task.higher_is_better(), dict)
#     assert task.aggregation().keys() == task.higher_is_better().keys()
#
#     for v in task.higher_is_better().values():
#         assert v in [True, False]
#
#     assert isinstance(task.VERSION, int)
#
#     # test deterministic docs
#     # (don't test train because it's slow)
#
#     task2 = task_class()
#
#     limit = None
#
#     if taskname in ["triviaqa"] or taskname.startswith("pile_"):
#         limit = 10000
#     if task.has_validation_docs():
#         arr = list(islice(task.validation_docs(), limit))
#         arr2 = list(islice(task2.validation_docs(), limit))
#
#         assert arr == arr2
#
#         reqs = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
#         reqs2 = [task2.construct_requests(doc, task2.doc_to_text(doc)) for doc in arr2]
#
#         assert reqs == reqs2
#
#     if task.has_test_docs():
#         arr = list(islice(task.test_docs(), limit))
#         arr2 = list(islice(task2.test_docs(), limit))
#
#         assert arr == arr2
#
#         reqs = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
#         reqs2 = [task2.construct_requests(doc, task2.doc_to_text(doc)) for doc in arr2]
#
#         assert reqs == reqs2
#
#     if task.has_training_docs():
#         arr = list(islice(task.training_docs(), limit))
#         arr2 = list(islice(task2.training_docs(), limit))
#
#         assert arr == arr2
#
#         reqs = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
#         reqs2 = [task2.construct_requests(doc, task2.doc_to_text(doc)) for doc in arr2]
#
#         assert reqs == reqs2
#
#
# @pytest.mark.parametrize("taskname,task_class", tasks.TASK_REGISTRY.items())
# def test_documents_and_requests(taskname, task_class):
#     print("Evaluating task", taskname)
#     task = task_class()
#     fns = []
#     if task.has_training_docs():
#         fns.append(task.training_docs)
#     if task.has_validation_docs():
#         fns.append(task.validation_docs)
#     # test doc might not have labels
#     # if task.has_test_docs(): fns.append(task.test_docs)
#
#     for fn in fns:
#         # print(list(islice(fn(), 10)))
#         for doc in islice(fn(), 10):
#
#             txt = task.doc_to_text(doc)
#             tgt = task.doc_to_target(doc)
#
#             assert isinstance(txt, str)
#             assert isinstance(tgt, str)
#
#             # space convention
#             # allow txt to have length 0 for perplexity-like tasks since the model tacks an <|endoftext|> on
#             if len(txt) != 0:
#                 assert txt[-1] != " "
#                 assert tgt[0] == " " or txt[-1] == "\n"
#
#             reqs = task.construct_requests(doc, txt)
#
#             # construct_requests can return just one request
#             if not isinstance(reqs, (list, tuple)):
#                 reqs = [reqs]
#
#             # todo: mock lm after refactoring evaluator.py to not be a mess
#             # for req in reqs:
#             #     assert isinstance(req, base.Request)
