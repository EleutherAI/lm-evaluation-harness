# import lm_eval.base as base
from typing import List

import pytest

# import lm_eval.models as models
import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks


# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


@pytest.mark.parametrize(
    "task_name,limit,model,model_args",
    [
        (
            ["arc_easy"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
        )
    ],
)
def test_evaluator(task_name: List[str], limit: int, model: str, model_args: str):
    task_name = task_name
    limit = 10

    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
    )
    assert e1 is not None

    lm = api.registry.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )
    task_manager = tasks.TaskManager()
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
