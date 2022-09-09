import random
import pytest
import os
import json
import hashlib
import collections

import lm_eval
from lm_eval.api.utils import DEFAULT_SEED, set_seed


def _assert_target(name, ob):
    fname = f"tests/testdata/{name}.json"
    if os.path.exists(fname):
        with open(fname) as fh:
            # Use relative tolerance of 1e-5 and absolute tolerance of 1e-8
            # assuming most metrics work on `float32` values, which is the common
            # default floating type across popular libraries (PyTorch, Tensorflow, and JAX).
            assert _flatten(json.load(fh)) == pytest.approx(
                _flatten(json.loads(json.dumps(ob, sort_keys=True))), rel=1e-5, abs=1e-8
            )
    else:
        with open(fname, "w") as fh:
            json.dump(ob, fh, sort_keys=True)


def _assert_target_hashed(name, ob):
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
def _flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Make sure eval results for a task version are stable


@pytest.mark.skip(reason="Version stability are not setup for `PropmtSourceTask`s")
# @pytest.mark.parametrize("task_name,task_class", tasks.TASK_REGISTRY.items())
def test_versions_stable(task_name, task_class):
    set_seed()
    os.makedirs("tests/testdata", exist_ok=True)
    task = lm_eval.get_task(task_name)
    model = lm_eval.get_model("dummy")

    def ll_fn(requests):
        for ctx, cont in requests:
            if len(ctx) == 0:
                continue
            # Space convention
            assert ctx[-1] != " "
            assert cont[0] == " " or ctx[-1] == "\n"

        _assert_target_hashed(
            f"{task_name}-v{task_class.VERSION}-loglikelihood", requests
        )
        res = []

        random.seed(DEFAULT_SEED)
        for _ in requests:
            res.append((-random.random(), False))

        return res

    def ll_perp_fn(requests):
        for (string,) in requests:
            assert isinstance(string, str)

        _assert_target_hashed(
            f"{task_name}-v{task_class.VERSION}-loglikelihood_rolling", requests
        )
        res = []

        random.seed(DEFAULT_SEED)
        for _ in requests:
            res.append(-random.random())

        return res

    def greedy_until(requests):
        res = []
        _assert_target_hashed(
            f"{task_name}-v{task_class.VERSION}-greedy_until", requests
        )

        for ctx, _ in requests:
            res.append("none")
            assert ctx.strip() != ""

        return res

    model.loglikelihood = ll_fn
    model.loglikelihood_rolling = ll_perp_fn
    model.greedy_until = greedy_until

    limit = None
    result = lm_eval.evaluate(
        model=model,
        tasks=[task],
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=10,
    )

    _assert_target(f"{task_name}-v{task_class.VERSION}-res", result)
