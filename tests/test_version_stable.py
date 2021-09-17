import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import random
import pytest
import os
import json
import hashlib


os.makedirs("tests/testdata", exist_ok=True)


def assert_target(name, ob):
    fname = f"tests/testdata/{name}.json"
    if os.path.exists(fname):
        with open(fname) as fh:
            assert json.load(fh) == json.loads(json.dumps(ob, sort_keys=True))
    else:
        with open(fname, 'w') as fh:
            json.dump(ob, fh, sort_keys=True)

def assert_target_hashed(name, ob):
    fname = f"tests/testdata/{name}"
    if os.path.exists(fname):
        with open(fname) as fh:
            assert fh.read() == hashlib.sha256(json.dumps(ob, sort_keys=True).encode('utf-8')).hexdigest()
    else:
        with open(fname, 'w') as fh:
            fh.write(hashlib.sha256(json.dumps(ob, sort_keys=True).encode('utf-8')).hexdigest())


# make sure eval results for a task version are stable

@pytest.mark.parametrize("taskname,Task", tasks.TASK_REGISTRY.items())
def test_versions_stable(taskname, Task):
    task_dict = tasks.get_task_dict([taskname])
    lm = models.get_model('dummy')()

    def ll_fn(reqs):
        for ctx, cont in reqs:
            if len(ctx) == 0: continue
            # space convention
            assert ctx[-1] != ' '
            assert cont[0] == ' ' or ctx[-1] == '\n'
        
        assert_target_hashed(f"{taskname}-v{Task.VERSION}-loglikelihood", reqs)
        res = []
        
        random.seed(42)
        for _ in reqs:
            res.append((-random.random(), False))

        return res

    def ll_perp_fn(reqs):
        for string, in reqs:
            assert isinstance(string, str)

        assert_target_hashed(f"{taskname}-v{Task.VERSION}-loglikelihood_rolling", reqs)
        res = []

        random.seed(42)
        for _ in reqs:
            res.append(-random.random())

        return res
    
    def greedy_until(reqs):
        res = []
        assert_target_hashed(f"{taskname}-v{Task.VERSION}-greedy_until", reqs)
        
        for ctx, _ in reqs:
            res.append("lol")
            assert ctx.strip() != ''

        return res

    lm.loglikelihood = ll_fn
    lm.loglikelihood_rolling = ll_perp_fn
    lm.greedy_until = greedy_until

    limit = None
    res = evaluator.evaluate(lm, task_dict, False, 0, limit, bootstrap_iters=10)
    assert_target(f"{taskname}-v{Task.VERSION}-res", res)
