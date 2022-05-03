import os
import tempfile
import lm_eval.base as base
import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import random
import pytest


# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


@pytest.mark.parametrize("taskname", tasks.TASK_REGISTRY.keys())
def test_evaluator(taskname):
    task_dict = tasks.get_task_dict([taskname])

    with tempfile.NamedTemporaryFile() as tfile:
        lm = base.CachingLM(models.get_model("dummy")(), tfile.name)

        def ll_fn(reqs):
            for ctx, cont in reqs:
                if len(ctx) == 0:
                    continue
                # space convention
                assert ctx[-1] != " "
                assert cont[0] == " " or ctx[-1] == "\n"

            res = []

            random.seed(42)
            for _ in reqs:
                res.append((-random.random(), False))

            return res

        def ll_perp_fn(reqs):
            for (string,) in reqs:
                assert isinstance(string, str)

            res = []
            random.seed(42)
            for _ in reqs:
                res.append(-random.random())

            return res

        lm.loglikelihood = ll_fn
        lm.loglikelihood_rolling = ll_perp_fn

        limit = 10
        e1 = evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=0,
            limit=limit,
            bootstrap_iters=10,
            description_dict=None,
        )
        e2 = evaluator.evaluate(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=0,
            limit=limit,
            bootstrap_iters=10,
            description_dict=None,
        )

        # check that caching is working
        assert e1 == e2
