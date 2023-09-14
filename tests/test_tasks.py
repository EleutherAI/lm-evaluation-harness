from itertools import islice
import pytest
from .utils import new_tasks
import lm_eval.tasks as tasks
from lm_eval.api.task import ConfigurableTask


# Default Task
TASKS = ["arc_easy"]


def task_class():
    global TASKS
    # CI: new_tasks checks if any modifications have been made
    task_classes = new_tasks()
    # Check if task_classes is empty
    if task_classes:
        return [tasks.TASK_REGISTRY.get(x)() for x in task_classes]
    else:
        return [tasks.TASK_REGISTRY.get(x)() for x in TASKS]


@pytest.fixture()
def limit() -> int:
    return 10


# Tests
@pytest.mark.parametrize("task_class", task_class())
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
            # assert all(len(x) == 4 for x in _array)
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
            assert all(isinstance(label, int) for label in _array_target)
        # _array_text = [task.doc_to_text(doc) for doc in arr]
        # Not working
        # assert all(tgt[0] == " " or txt[-1] == "\n" if  len(txt) != 0 else True for txt, tgt in zip(_array_text, _array_target))

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
        # assert all(isinstance(doc, list) for doc in requests)
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
