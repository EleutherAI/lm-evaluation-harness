import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import random
import pytest


# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces

@pytest.mark.parametrize("taskname,Task", tasks.TASK_REGISTRY.items())
def test_evaluator(taskname, Task):
    task_dict = tasks.get_task_dict([taskname])
    lm = models.get_model('dummy')()

    def ll_fn(reqs):
        for ctx, cont in reqs:
            if len(ctx) == 0: continue
            # space convention
            assert ctx[-1] != ' '
            assert cont[0] == ' ' or ctx[-1] == '\n'
        
        res = []
        
        random.seed(42)
        for _ in reqs:
            res.append((-random.random(), False))

        return res

    def ll_perp_fn(reqs):
        for string, in reqs:
            assert isinstance(string, str)

        res = []
        random.seed(42)
        for _ in reqs:
            res.append(-random.random())

        return res

    lm.loglikelihood = ll_fn
    lm.loglikelihood_rolling = ll_perp_fn
    evaluator.evaluate(lm, task_dict, False, 0, 10, bootstrap_iters=10)
