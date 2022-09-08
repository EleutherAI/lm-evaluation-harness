import os
import random
import pytest

import lm_eval
import lm_eval.tasks as tasks
import lm_eval.api.model as model
import lm_eval.models as models
import lm_eval.evaluator as evaluator
from lm_eval.api.utils import DEFAULT_SEED, set_seed


# TODO: More fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


def _ll_fn(requests):
    for ctx, cont in requests:
        if len(ctx) == 0:
            continue
        # Check text-target-separator default spacing convention.
        # ctx + (' ' + cont)
        assert ctx[-1] != " "
        assert cont[0] == " "
    res = []
    random.seed(DEFAULT_SEED)
    for _ in requests:
        res.append((-random.random(), False))
    return res


def _ll_perp_fn(requests):
    for (string,) in requests:
        assert isinstance(string, str)
    res = []
    random.seed(DEFAULT_SEED)
    for _ in requests:
        res.append(-random.random())
    return res


@pytest.mark.parametrize("task_name", lm_eval.list_tasks())
def test_evaluator(task_name):
    set_seed()
    template_names = tasks.list_templates(task_name)
    # Only choose 1 promptsource template.
    template_name = template_names[0] if template_names else None
    task = tasks.get_task(task_name, template_name)

    os.system("rm test_cache.db")
    lm = model.CachingLM(models.get_model("dummy"), "test_cache.db")
    lm.loglikelihood = _ll_fn
    lm.loglikelihood_rolling = _ll_perp_fn

    limit = 5
    e1 = evaluator.evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        bootstrap_iters=10,
        limit=limit,
    )["results"]
    e2 = evaluator.evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=0,
        bootstrap_iters=10,
        limit=limit,
    )["results"]
    # Check that caching is working
    assert e1 == e2
