import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import random
import pytest
import os
import json
import hashlib
import collections


os.makedirs("tests/testdata", exist_ok=True)


def assert_target(name, ob):
    fname = f"tests/testdata/{name}.json"
    if os.path.exists(fname):
        with open(fname) as fh:
            # Use relative tolerance of 1e-5 and absolute tolerance of 1e-8
            # assuming most metrics work on `float32` values, which is the common
            # default floating type across popular libraries (PyTorch, Tensorflow, and JAX).
            assert flatten(json.load(fh)) == pytest.approx(
                flatten(json.loads(json.dumps(ob, sort_keys=True))), rel=1e-5, abs=1e-8
            )
    else:
        with open(fname, "w") as fh:
            json.dump(ob, fh, sort_keys=True)


def assert_target_hashed(name, ob):
    fname = f"tests/testdata/{name}"
    if os.path.exists(fname):
        with open(fname) as fh:
            assert (
                fh.read()
                == hashlib.sha256(
                    json.dumps(ob, sort_keys=True).encode("utf-8")
                ).hexdigest()
            )
    else:
        with open(fname, "w") as fh:
            fh.write(
                hashlib.sha256(
                    json.dumps(ob, sort_keys=True).encode("utf-8")
                ).hexdigest()
            )


# from https://stackoverflow.com/a/6027615
def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# make sure eval results for a task version are stable


@pytest.mark.parametrize("taskname,task_class", tasks.TASK_REGISTRY.items())
def test_versions_stable(taskname, task_class):
    task_dict = tasks.get_task_dict([taskname])
    lm = models.get_model("dummy")()

    def ll_fn(reqs):
        for ctx, cont in reqs:
            if len(ctx) == 0:
                continue
            # space convention
            assert ctx[-1] != " "
            assert cont[0] == " " or ctx[-1] == "\n"

        assert_target_hashed(f"{taskname}-v{task_class.VERSION}-loglikelihood", reqs)
        res = []

        random.seed(42)
        for _ in reqs:
            res.append((-random.random(), False))

        return res

    def ll_perp_fn(reqs):
        for (string,) in reqs:
            assert isinstance(string, str)

        assert_target_hashed(
            f"{taskname}-v{task_class.VERSION}-loglikelihood_rolling", reqs
        )
        res = []

        random.seed(42)
        for _ in reqs:
            res.append(-random.random())

        return res

    def greedy_until(reqs):
        res = []
        assert_target_hashed(f"{taskname}-v{task_class.VERSION}-greedy_until", reqs)

        for ctx, _ in reqs:
            res.append("lol")
            assert ctx.strip() != ""

        return res

    lm.loglikelihood = ll_fn
    lm.loglikelihood_rolling = ll_perp_fn
    lm.greedy_until = greedy_until

    limit = None
    result = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=10,
        description_dict=None,
    )

    assert_target(f"{taskname}-v{task_class.VERSION}-res", result)
