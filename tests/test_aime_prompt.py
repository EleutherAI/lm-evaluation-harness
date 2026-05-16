"""Regression test: AIME tasks must produce non-empty prompts for API backends."""

import pytest

from lm_eval.tasks import TaskManager, get_task_dict


@pytest.mark.parametrize("task_name", ["aime24", "aime", "aime25"])
def test_aime_prompt_not_empty(task_name):
    """Ensure AIME doc_to_text never returns empty prompt (fixes API backend errors)."""
    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)
    task = task_dict[task_name]
    doc = next(iter(task.test_docs()))
    prompt = task.doc_to_text(doc)
    assert isinstance(prompt, str), "doc_to_text must return a string"
    assert prompt.strip() != "", "Prompt must be non-empty for API backends"
