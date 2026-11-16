"""Shared pytest fixtures for lm-eval tests."""

import os
from unittest.mock import Mock

import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import FewshotConfig, TaskConfig


@pytest.fixture
def on_ci():
    """True when running in GitHub Actions CI."""
    return os.environ.get("GITHUB_ACTIONS") == "true"


@pytest.fixture
def fewshot_config():
    """Default FewshotConfig with standard values."""
    return FewshotConfig(
        sampler="default",
        split="train",
        fewshot_delimiter="\n\n",
        target_delimiter=" ",
    )


@pytest.fixture
def task_config():
    """Default TaskConfig with standard values.

    Also initializes fewshot_config via __post_init__.
    """
    return TaskConfig(
        task="test_task",
        dataset_path="test_dataset",
        test_split="test",
        fewshot_split="train",
        doc_to_text="question",
        doc_to_target="answer",
        target_delimiter=" ",
        fewshot_delimiter="\n\n",
    )


@pytest.fixture
def mock_configurable_task(task_config):
    """Mock ConfigurableTask with real TaskConfig (and FewshotConfig via __post_init__)."""
    task = Mock(spec=ConfigurableTask)

    # Use real TaskConfig (initializes fewshot_config in __post_init__)
    task.config = task_config
    task.fewshot_cfg = task_config.fewshot_config

    # Default attributes
    task.multiple_input = False

    # Mock methods - use real build_qa_turn
    task.build_qa_turn = lambda **kwargs: ConfigurableTask.build_qa_turn(task, **kwargs)
    task.resolve_field = Mock(return_value=None)

    # Mock sampler
    task.sampler = Mock()
    task.sampler.sample = Mock(return_value=[])

    return task
