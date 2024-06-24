import os

import pytest

import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks


@pytest.mark.parametrize(
    "limit,model,model_args",
    [
        (
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
        ),
    ],
)
def test_include_correctness(limit: int, model: str, model_args: str):
    task_name = ["arc_easy"]

    task_manager = tasks.TaskManager()
    task_dict = tasks.get_task_dict(task_name, task_manager)

    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
    )
    assert e1 is not None

    # run with evaluate() and "arc_easy" test config (included from ./testconfigs path)
    lm = api.registry.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )

    task_name = ["arc_easy"]

    task_manager = tasks.TaskManager(
        include_path=os.path.dirname(os.path.abspath(__file__)) + "/testconfigs",
        include_defaults=False,
    )
    task_dict = tasks.get_task_dict(task_name, task_manager)

    e2 = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
    )

    assert e2 is not None
    # check that caching is working

    def r(x):
        return x["results"]["arc_easy"]

    assert all(
        x == y
        for x, y in zip([y for _, y in r(e1).items()], [y for _, y in r(e2).items()])
    )


# test that setting include_defaults = False works as expected and that include_path works
def test_no_include_defaults():
    task_name = ["arc_easy"]

    task_manager = tasks.TaskManager(
        include_path=os.path.dirname(os.path.abspath(__file__)) + "/testconfigs",
        include_defaults=False,
    )
    # should succeed, because we've included an 'arc_easy' task from this dir
    task_dict = tasks.get_task_dict(task_name, task_manager)

    # should fail, since ./testconfigs has no arc_challenge task
    task_name = ["arc_challenge"]
    with pytest.raises(KeyError):
        task_dict = tasks.get_task_dict(task_name, task_manager)  # noqa: F841


# test that include_path containing a task shadowing another task's name fails
# def test_shadowed_name_fails():

#     task_name = ["arc_easy"]

#     task_manager = tasks.TaskManager(include_path=os.path.dirname(os.path.abspath(__file__)) + "/testconfigs")
#     task_dict = tasks.get_task_dict(task_name, task_manager)
