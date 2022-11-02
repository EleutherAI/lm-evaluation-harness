import logging
import pytest
import numpy as np
from typing import Optional, Tuple
from itertools import islice
from promptsource.templates import Template

import lm_eval.tasks as tasks
from lm_eval.api.task import Task
from lm_eval.api.request import Request
from lm_eval.api.utils import set_seed, DEFAULT_SEED


logger = logging.getLogger(__name__)


def _get_deterministic_template(
    task_name: str,
) -> Tuple[Optional[Template], bool]:
    """Some `promptsource` templates randomize the ordering of prompt attributes
    If `task_class` does not have a prompt template with non-random ordering we
    return None.

    Returns:
        (prompt template, is_deterministic)
    """
    # Only choose 1 promptsource template.
    prompt_template = None
    templates = tasks.get_templates(task_name)
    if templates.all_template_names:
        for template_name in templates.all_template_names:
            prompt_template = templates[template_name]
            # Hacky way to ensure we only grab a deterministic jinja template.
            if (
                "range(" not in prompt_template.jinja
                and "random" not in prompt_template.jinja
            ):
                return prompt_template, True
        # Return the last non-deterministic template.
        return prompt_template, False
    return None, False


def _filter_docs(task: Task):
    def _filter(doc: dict):
        return not task.invalid_doc_for_prompt(doc)

    return _filter


@pytest.mark.parametrize("task_name,task_class", tasks.TASK_REGISTRY.items())
def test_basic_interface(task_name: str, task_class: Task):
    logger.info("Evaluating task", task_name)
    prompt_template, is_deterministic = _get_deterministic_template(task_name)
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
def test_documents_and_requests(task_name: str, task_class: Task):
    set_seed()
    logger.info("Evaluating task", task_name)
    prompt_template, _ = _get_deterministic_template(task_name)
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


def test_arg_string_task_creation():
    import itertools

    TEST_EXAMPLE_SEPS = [
        # Test `=` symbol in value string
        "\n===TEST_SEPARATOR===\n",
        # Test whitespace only separators
        " ",
        " \t\t  ",
        "\n\n\n\n",
        # Test empty string separator
        "",
        # Test misc. symbols in separator
        "[[[[]]]]",
        "<<___>>",
        "(())",
    ]
    TEST_TEXT_TARGET_SEPS = [
        # Test whitespace separators
        "   ",
        " \t ",
        "\n\n\n",
    ]

    # Ensure parsing properly handles args.
    for example_sep, text_target_sep in itertools.product(
        TEST_EXAMPLE_SEPS, TEST_TEXT_TARGET_SEPS
    ):
        test_arg_string = f" save_examples=False,example_separator={example_sep},text_target_separator={text_target_sep}"
        task = tasks.get_task_list_from_args_string(
            "wnli",
            template_names=["confident"],
            task_args=test_arg_string,
        )[0]

        assert task.save_examples is False
        assert task.example_separator == example_sep
        assert task.text_target_separator == text_target_sep

    # Ensure fewshot context is formatted as expected.
    TEST_EXAMPLE_SEP = "\n===TEST_SEPARATOR===\n"
    TEST_TEXT_TARGET_SEP = "   "
    test_arg_string = f" save_examples=False,example_separator={TEST_EXAMPLE_SEP},text_target_separator={TEST_TEXT_TARGET_SEP}"
    task = tasks.get_task_list_from_args_string(
        "wnli",
        template_names=["confident"],
        task_args=test_arg_string,
    )[0]
    context = task.fewshot_context(
        task.validation_docs()[0],
        num_fewshot=2,
        rng=np.random.default_rng(DEFAULT_SEED),
    )[0]
    expected = f"""If it's true that
The man couldn't lift his son because he was so heavy.
how confident should I be that
The man was so heavy.
very confident or not confident?   not confident
===TEST_SEPARATOR===
If it's true that
As Ollie carried Tommy up the long winding steps, his legs ached.
how confident should I be that
Ollie's legs ached.
very confident or not confident?   very confident
===TEST_SEPARATOR===
If it's true that
The drain is clogged with hair. It has to be cleaned.
how confident should I be that
The hair has to be cleaned.
very confident or not confident?"""
    assert context == expected

    # Ensure tasks don't instantiate with invalid args.
    with pytest.raises(AssertionError):
        bad_save_examples_arg_string = "example_separator=\t,save_examples=yes"
        task = tasks.get_task_list_from_args_string(
            "wnli",
            template_names=["confident"],
            task_args=bad_save_examples_arg_string,
        )[0]

        bad_example_sep_arg_string = "example_separator=False,save_examples=False"
        task = tasks.get_task_list_from_args_string(
            "wnli",
            template_names=["confident"],
            task_args=bad_example_sep_arg_string,
        )[0]

        bad_text_sep_arg_string = "text_target_separator=___"
        task = tasks.get_task_list_from_args_string(
            "wnli",
            template_names=["confident"],
            task_args=bad_text_sep_arg_string,
        )[0]
