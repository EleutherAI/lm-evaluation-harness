import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig


class MockTargetTask(ConfigurableTask):
    def __init__(self, target_type="auto"):
        self._config = TaskConfig(
            task="mock_target",
            output_type="generate_until",
            doc_to_target="{{target}}",
            doc_to_target_type=target_type,
        )
        self.prompt = None
        self.features = set()


def test_doc_to_target_auto_preserves_legacy_list_parsing():
    task = MockTargetTask()

    assert task.doc_to_target({"target": "[1, 2, 3]"}) == [1, 2, 3]


def test_doc_to_target_string_preserves_literal_list_text():
    task = MockTargetTask(target_type="string")

    assert task.doc_to_target({"target": "[1, 2, 3]"}) == "[1, 2, 3]"


def test_doc_to_target_list_parses_literal_list_text():
    task = MockTargetTask(target_type="list")

    assert task.doc_to_target({"target": "[1, 2, 3]"}) == [1, 2, 3]


def test_doc_to_target_rejects_unknown_target_type():
    with pytest.raises(ValueError, match="doc_to_target_type"):
        MockTargetTask(target_type="mapping")


def test_doc_to_target_list_rejects_non_list_literal():
    task = MockTargetTask(target_type="list")

    with pytest.raises(ValueError, match="list literal"):
        task.doc_to_target({"target": "answer"})
