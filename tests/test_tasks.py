import pytest
import numpy as np
from typing import Optional, Tuple
from itertools import islice
from promptsource.templates import DatasetTemplates, Template

import lm_eval.tasks as tasks
from lm_eval.api.task import PromptSourceTask
from lm_eval.api.request import Request
from lm_eval.api.utils import set_seed


_SEED = 42


def _get_deterministic_template(
    task_class: PromptSourceTask,
) -> Tuple[Optional[Template], bool]:
    """Some `promptsource` templates randomize the ordering of prompt attributes
    If `task_class` does not have a prompt template with non-random ordering we
    return None.

    :return: (prompt template, is_deterministic)
    """
    # Only choose 1 promptsource template.
    prompt = None
    templates = tasks.get_task_templates(task_class)
    if templates.all_template_names:
        for prompt_name in templates.all_template_names:
            prompt = templates[prompt_name]
            # Hacky way to ensure we only grab a deterministic jinja template.
            if "range(" not in prompt.jinja and "random" not in prompt.jinja:
                return prompt, True
        # Return the last non-deterministic template.
        return prompt, False
    return None, False


def _filter_docs(task: PromptSourceTask):
    def _filter(doc: dict):
        return not task.invalid_doc_for_prompt(doc)

    return _filter


@pytest.mark.parametrize("task_name,task_class", tasks.TASK_REGISTRY.items())
def test_basic_interface(task_name: str, task_class: PromptSourceTask):
    print("Evaluating task", task_name)
    task_class = tasks.get_task(task_name)

    prompt_template, is_deterministic = _get_deterministic_template(task_class)
    task = task_class(prompt_template=prompt_template)

    assert task.has_training_docs() in [True, False]
    assert task.has_validation_docs() in [True, False]
    assert task.has_test_docs() in [True, False]

    assert isinstance(task.aggregation(), dict)
    assert isinstance(task.higher_is_better(), dict)
    assert task.aggregation().keys() == task.higher_is_better().keys()
    for v in task.higher_is_better().values():
        assert v in [True, False]
    assert isinstance(task.VERSION, int)

    # Test deterministic docs (NOTE: Don't test train because it's slow).
    # Return if the prompts are non-deterministic here.
    if not is_deterministic:
        return

    limit = 100
    task2 = task_class(prompt_template=prompt_template)

    if task.has_validation_docs():
        arr = list(task.validation_docs().filter(_filter_docs(task)))[:limit]
        arr2 = list(task2.validation_docs().filter(_filter_docs(task2)))[:limit]
        assert arr == arr2
        requests = [
            task.construct_requests(doc, task.doc_to_text(doc), {"num_fewshot": 0})
            for doc in arr
        ]
        requests2 = [
            task.construct_requests(doc, task2.doc_to_text(doc), {"num_fewshot": 0})
            for doc in arr2
        ]
        assert requests == requests2

    if task.has_test_docs():
        arr = list(task.test_docs().filter(_filter_docs(task)))[:limit]
        arr2 = list(task2.test_docs().filter(_filter_docs(task2)))[:limit]
        assert arr == arr2
        requests = [
            task.construct_requests(doc, task.doc_to_text(doc), {"num_fewshot": 0})
            for doc in arr
        ]
        requests2 = [
            task2.construct_requests(doc, task2.doc_to_text(doc), {"num_fewshot": 0})
            for doc in arr2
        ]
        assert requests == requests2


@pytest.mark.parametrize("task_name,task_class", tasks.TASK_REGISTRY.items())
def test_documents_and_requests(task_name: str, task_class: PromptSourceTask):
    print("Evaluating task", task_name)
    task_class = tasks.get_task(task_name)
    prompt_template, _ = _get_deterministic_template(task_class)
    task = task_class(prompt_template=prompt_template)

    fns = []

    # Training docs are too expensive to run on CI.
    # if task.has_training_docs():
    #     fns.append(task.training_docs)
    if task.has_validation_docs():
        fns.append(task.validation_docs)
    for fn in fns:
        docs = fn().filter(_filter_docs(task))
        for doc in islice(docs, 5):
            text = task.doc_to_text(doc)
            target = task.doc_to_target(doc)

            assert isinstance(text, str)
            assert isinstance(target, list)

            requests = task.construct_requests(doc, text, {"num_fewshot": 0})

            # Construct_requests can return just one request
            if not isinstance(requests, (list, tuple)):
                requests = [requests]
            # TODO: Mock lm after refactoring evaluator.py to not be a mess
            for req in requests:
                assert isinstance(req, Request)


def test_descriptions():
    rng = np.random.default_rng(_SEED)
    num_fewshot = 1

    task_to_template = {
        "axg": DatasetTemplates("super_glue", "axg")["can we infer"],
        "wnli": DatasetTemplates("glue", "wnli")["confident"],
    }
    task_to_description = {
        "axg": "This task is used to measure  measure gender bias in coreference "
        "resolution systems. Follow the prompt instructions to complete the task",
        "wnli": "This task tests reading comprehension. Follow the prompt "
        "instructions to complete the task.",
    }

    task_dict = {
        task: tasks.get_task(task)(
            prompt_template=template, description=task_to_description[task]
        )
        for task, template in task_to_template.items()
    }
    for _, task in task_dict.items():
        docs = task.evaluation_docs()
        for _, doc in (
            zip(range(num_fewshot), docs) if num_fewshot > 0 else enumerate(docs)
        ):
            ctx = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                rng=rng,
            )[0]
            assert task.description in ctx
