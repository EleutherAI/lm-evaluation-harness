import os
import numpy as np
import random
import pytest

import lm_eval.tasks as tasks
import lm_eval.api.model as model
import lm_eval.models as models
import lm_eval.evaluator as evaluator
from lm_eval.api.task import PerplexityTask
from lm_eval.api.utils import set_seed


_SEED = 42


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
    random.seed(_SEED)
    for _ in requests:
        res.append((-random.random(), False))
    return res


def _ll_perp_fn(requests):
    for (string,) in requests:
        assert isinstance(string, str)
    res = []
    random.seed(_SEED)
    for _ in requests:
        res.append(-random.random())
    return res


@pytest.mark.parametrize("task_name,task_class", tasks.TASK_REGISTRY.items())
def test_evaluator(task_name, task_class):
    set_seed(_SEED)
    task_class = tasks.get_task(task_name)
    templates = tasks.get_task_templates(task_class)
    # Only choose 1 promptsource template.
    if templates.all_template_names:
        prompt_name = templates.all_template_names[0]
        prompt = templates[prompt_name]
        task_dict = {f"{task_name}+{prompt_name}": task_class(prompt_template=prompt)}
    elif issubclass(task_class, PerplexityTask):
        task_dict = {f"{task_name}+null": task_class()}
    else:
        assert False, "No templates for task"

    os.system("rm test_cache.db")
    lm = model.CachingLM(models.get_model("dummy")(), "test_cache.db")
    lm.loglikelihood = _ll_fn
    lm.loglikelihood_rolling = _ll_perp_fn

    limit = 5
    e1 = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=0,
        bootstrap_iters=10,
        limit=limit,
        rng=np.random.default_rng(_SEED),
    )["results"]
    e2 = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=0,
        bootstrap_iters=10,
        limit=limit,
        rng=np.random.default_rng(_SEED),
    )["results"]
    # Check that caching is working
    assert e1 == e2
