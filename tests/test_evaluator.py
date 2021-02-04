import lm_eval.tasks as tasks
import lm_eval.models as models
import lm_eval.evaluator as evaluator
import pytest


# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces

@pytest.mark.parametrize("taskname,Task", tasks.TASK_REGISTRY.items())
def test_evaluator(taskname, Task):
    task_dict = tasks.get_task_dict([taskname])
    lm = models.get_model('dummy')()
    evaluator.evaluate(lm, task_dict, False, 0, 10)