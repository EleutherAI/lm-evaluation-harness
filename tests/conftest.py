"""Shared pytest fixtures for lm-eval tests."""

import os
from unittest.mock import Mock

import pytest

from lm_eval.api.task import Task
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
def make_task():
    """Factory fixture that creates a real Task with minimal config and no dataset loading.

    Usage:
        task = make_task("my_task")
        task = make_task("my_task", output_type="multiple_choice")
        task = make_task("my_task", metric_list=[{"metric": "acc"}])
    """

    def _make(
        task_name: str = "test_task",
        output_type: str = "generate_until",
        n_eval_docs: int = 0,
        **config_overrides,
    ) -> Task:
        defaults = {
            "dataset_path": "dummy",
            "test_split": "test",
            "generation_kwargs": {
                "until": ["\n"],
                "temperature": 0,
                "do_sample": False,
            },
        }
        defaults.update(config_overrides)
        cfg = TaskConfig(
            task=task_name,
            output_type=output_type,  # type: ignore[arg-type]
            **defaults,
        )
        task = Task.from_config(cfg)
        task._dataset = {"test": [{}] * n_eval_docs}
        return task

    return _make


@pytest.fixture
def mock_configurable_task(task_config):
    """Mock Task with real TaskConfig (and FewshotConfig via __post_init__)."""
    task = Mock(spec=Task)

    # Use real TaskConfig (initializes fewshot_config in __post_init__)
    task.config = task_config
    task._fewshot_cfg = task_config.fewshot_config

    # Default attributes
    task._multiple_inputs = False

    # Mock methods - use real _build_qa_turn
    task._build_qa_turn = lambda **kwargs: Task._build_qa_turn(task, **kwargs)
    task._resolve_field = Mock(return_value=None)

    # Mock sampler
    task.sampler = Mock()
    task.sampler.sample = Mock(return_value=[])

    return task
