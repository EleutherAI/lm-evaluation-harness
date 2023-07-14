import os

# import lm_eval.base as base
import lm_eval.api.registry as registry
import lm_eval.tasks as tasks

# import lm_eval.models as models

import lm_eval.evaluator as evaluator
import random
import pytest


# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


@pytest.mark.parametrize(
    ("task_name,limit,model,model_args"),
    [
        (
            ["arc_easy"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
        )
    ],
)
def test_evaluator(task_name: list[str], limit: int, model: str, model_args: str):
    task_name = task_name
    limit = 10
    model, model_args = model, model_args
    # task_dict = tasks.get_task_dict(task)

    # TODO: re-add cachingLM
    # os.system("rm test_cache.db")
    # lm = base.CachingLM(models.get_model("dummy")(), "test_cache.db")
    # lm = registry.get_model("dummy")()

    # def ll_fn(reqs):
    #     for ctx, cont in [req.args for req in reqs]:
    #         if len(ctx) == 0:
    #             continue
    #         # space convention
    #         assert ctx[-1] != " "
    #         assert cont[0] == " " or ctx[-1] == "\n"
    #
    #     res = []
    #
    #     random.seed(42)
    #     for _ in reqs:
    #         res.append((-random.random(), False))
    #
    #     return res
    #
    # def ll_perp_fn(reqs):
    #     for (string,) in reqs:
    #         assert isinstance(string, str)
    #
    #     res = []
    #     random.seed(42)
    #     for _ in reqs:
    #         res.append(-random.random())
    #
    #     return res
    #
    # lm.loglikelihood = ll_fn
    # lm.loglikelihood_rolling = ll_perp_fn

    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
    )
    e2 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
    )

    # check that caching is working
    assert e1 == e2
