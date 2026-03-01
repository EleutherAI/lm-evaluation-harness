"""Task management for lm-evaluation-harness.

This module provides:
- TaskManager: Main class for discovering and loading evaluation tasks
- get_task_dict: Function to create a dictionary of task objects
- Helper functions for task name resolution
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import deprecated

# Import TaskManager
from .manager import TaskManager


if TYPE_CHECKING:
    from collections.abc import Mapping

    from lm_eval.api.task import Task


eval_logger = logging.getLogger(__name__)

__all__ = [
    "TaskManager",
    "get_task_dict",
    "get_task_name_from_config",
    "get_task_name_from_object",
]


@deprecated(
    "get_task_name_from_config is deprecated, and will be removed in a future version. Task names should be explicitly defined in task configs under the 'task' key."
)
def get_task_name_from_config(task_config: Mapping[str, str]) -> str:
    match task_config:
        case {"task": task_name}:
            return task_name
        case {"dataset_path": dataset_path, "dataset_name": dataset_name}:
            return f"{dataset_path}_{dataset_name}"
        case {"dataset_path": dataset_path}:
            return f"{dataset_path}"
        case _:
            raise ValueError(
                "Could not extract task name from config. Expected keys 'task' or 'dataset_path' (with optional 'dataset_name')."
            )


def get_task_name_from_object(task_object):
    if hasattr(task_object, "config"):
        return task_object.config["task"]

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def _log_task_dict(task_dict: dict, task_manager: TaskManager) -> None:
    """Log the selected tasks with hierarchy information."""
    from lm_eval.api.group import ConfigurableGroup
    from lm_eval.api.task import Task

    def pretty_print_task(task_name: str, indent: int):
        entry = task_manager.task_index.get(task_name)
        if entry and entry.yaml_path:
            yaml_path = Path(entry.yaml_path)
            lm_eval_tasks_path = Path(__file__).parent
            try:
                display_path = yaml_path.relative_to(lm_eval_tasks_path)
            except ValueError:
                display_path = yaml_path
        else:
            display_path = "N/A"
        pad = "  " * indent
        eval_logger.info(f"{pad}Task: {task_name} ({display_path})")

    def _log_nested(d: dict, indent: int = 0) -> None:
        for key, value in d.items():
            if isinstance(key, ConfigurableGroup):
                pad = "  " * indent
                label = "Group" if indent == 0 else "Subgroup"
                eval_logger.info(f"{pad}{label}: {key.group}")
                if isinstance(value, dict):
                    _log_nested(value, indent + 1)
                else:
                    eval_logger.info(f"{pad}  {key}: {value}")
            elif isinstance(key, str) and isinstance(value, Task):
                pretty_print_task(key, indent)
            else:
                eval_logger.info(f"{'  ' * indent}{key}: {value}")

    eval_logger.info("Selected tasks:")
    _log_nested(task_dict)


@deprecated("get_task_dict is deprecated. Use TaskManager.load() instead.")
def get_task_dict(
    task_name_list: str | list[str | dict | Task],
    task_manager: TaskManager | None = None,
):
    from lm_eval.api.task import Task

    """Creates a dictionary of task objects from either a name of task, config, or prepared Task object.

    :param task_name_list: List[Union[str, Dict, Task]]
        Name of model or LM object, see lm_eval.models.get_model
    :param task_manager: TaskManager = None
        A TaskManager object that stores indexed tasks. If not set,
        task_manager will load one. This should be set by the user
        if there are additional paths that want to be included
        via `include_path`

    :return
        Dictionary of task objects
    """
    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]

    if task_manager is None:
        task_manager = TaskManager()

    # Separate pre-built Task objects from specs (str/dict)
    specs = [s for s in task_name_list if isinstance(s, (str, dict))]
    task_objects = [s for s in task_name_list if isinstance(s, Task)]

    # Load all string/dict specs through load_task_or_group
    result = task_manager.load_task_or_group(specs) if specs else {}

    # Add pre-built Task objects directly
    for task_obj in task_objects:
        result[get_task_name_from_object(task_obj)] = task_obj

    # Log
    _log_task_dict(result, task_manager)

    return result
