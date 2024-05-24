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
    "task_name,limit,model,model_args,bootstrap_iters",
    [
        (
            ["arc_easy"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            0,
        ),
        (
            ["mmlu_abstract_algebra"],
            None,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            10000,
        ),
    ],
)
def test_evaluator(
    task_name: List[str], limit: int, model: str, model_args: str, bootstrap_iters: int
):
    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=bootstrap_iters,
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
        bootstrap_iters=bootstrap_iters,
    )

    assert e2 is not None
    # check that caching is working

    def r(x):
        if "arc_easy" in x["results"]:
            return x["results"]["arc_easy"]
        else:
            return x["results"]["mmlu_abstract_algebra"]

    assert all(
        x == y
        for x, y in zip([y for _, y in r(e1).items()], [y for _, y in r(e2).items()])
    )
