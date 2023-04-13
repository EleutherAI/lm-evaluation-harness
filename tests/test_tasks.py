import lm_eval.tasks as tasks
import lm_eval.base as base
import pytest
from itertools import islice


@pytest.mark.parametrize("taskname,task_class", tasks.TASK_REGISTRY.items())
def test_basic_interface(taskname, task_class):
    print("Evaluating task", taskname)
    task = task_class()

    assert task.has_training_docs() in [True, False]
    assert task.has_validation_docs() in [True, False]
    assert task.has_test_docs() in [True, False]

    assert isinstance(task.aggregation(), dict)
    assert isinstance(task.higher_is_better(), dict)
    assert task.aggregation().keys() == task.higher_is_better().keys()

    for v in task.higher_is_better().values():
        assert v in [True, False]

    assert isinstance(task.VERSION, int)

    # test deterministic docs
    # (don't test train because it's slow)

    task2 = task_class()

    limit = None

    if taskname in ["triviaqa"] or taskname.startswith("pile_"):
        limit = 10000
    if task.has_validation_docs():
        arr = list(islice(task.validation_docs(), limit))
        arr2 = list(islice(task2.validation_docs(), limit))

        assert arr == arr2

        reqs = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
        reqs2 = [task2.construct_requests(doc, task2.doc_to_text(doc)) for doc in arr2]

        assert reqs == reqs2

    if task.has_test_docs():
        arr = list(islice(task.test_docs(), limit))
        arr2 = list(islice(task2.test_docs(), limit))

        assert arr == arr2

        reqs = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
        reqs2 = [task2.construct_requests(doc, task2.doc_to_text(doc)) for doc in arr2]

        assert reqs == reqs2

    if task.has_training_docs():
        arr = list(islice(task.training_docs(), limit))
        arr2 = list(islice(task2.training_docs(), limit))

        assert arr == arr2

        reqs = [task.construct_requests(doc, task.doc_to_text(doc)) for doc in arr]
        reqs2 = [task2.construct_requests(doc, task2.doc_to_text(doc)) for doc in arr2]

        assert reqs == reqs2


@pytest.mark.parametrize("taskname,task_class", tasks.TASK_REGISTRY.items())
def test_documents_and_requests(taskname, task_class):
    print("Evaluating task", taskname)
    task = task_class()
    fns = []
    if task.has_training_docs():
        fns.append(task.training_docs)
    if task.has_validation_docs():
        fns.append(task.validation_docs)
    # test doc might not have labels
    # if task.has_test_docs(): fns.append(task.test_docs)

    for fn in fns:
        # print(list(islice(fn(), 10)))
        for doc in islice(fn(), 10):

            txt = task.doc_to_text(doc)
            tgt = task.doc_to_target(doc)

            assert isinstance(txt, str)
            assert isinstance(tgt, str)

            # space convention
            # allow txt to have length 0 for perplexity-like tasks since the model tacks an <|endoftext|> on
            if len(txt) != 0:
                assert txt[-1] != " "
                assert tgt[0] == " " or txt[-1] == "\n"

            reqs = task.construct_requests(doc, txt)

            # construct_requests can return just one request
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]

            # todo: mock lm after refactoring evaluator.py to not be a mess
            for req in reqs:
                assert isinstance(req, base.Request)
