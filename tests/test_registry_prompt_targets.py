from types import SimpleNamespace

import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.prompts import get_prompt


@pytest.mark.parametrize("prompt_name", ["question-newline-answer", "q-newline-a"])
def test_registry_prompt_preserves_gold_target_and_choices(prompt_name):
    task = ConfigurableTask.__new__(ConfigurableTask)
    task._config = SimpleNamespace(
        doc_to_text="question", doc_to_target="answer", doc_to_choice="choices"
    )
    task.features = ["question", "answer", "choices"]
    task.prompt = get_prompt(f"qa-basic:{prompt_name}")
    doc = {"question": "What is 1+1?", "answer": 1, "choices": ["1", "2"]}
    assert "What is 1+1?" in task.doc_to_text(doc)
    assert task.doc_to_target(doc) == 1
    assert task.doc_to_choice(doc) == ["1", "2"]
    assert task.doc_to_target(doc, doc_to_target=lambda d: "override") == "override"


def test_promptsource_object_still_supplies_targets_and_choices():
    task = ConfigurableTask.__new__(ConfigurableTask)
    task.features = []
    task.prompt = SimpleNamespace(
        apply=lambda d: ["question", "answer"],
        get_answer_choices_list=lambda d: ["answer", "other"],
    )
    assert task.doc_to_text({}) == "question"
    assert task.doc_to_target({}) == "answer"
    assert task.doc_to_choice({}) == ["answer", "other"]
