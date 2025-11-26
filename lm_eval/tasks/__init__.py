"""Task management for lm-evaluation-harness.

This module provides:
- TaskManager: Main class for discovering and loading evaluation tasks
- get_task_dict: Function to create a dictionary of task objects
- Helper functions for task name resolution
"""

from typing import Dict, List, Optional, Union

from lm_eval.api.task import Task
from lm_eval.evaluator_utils import get_subtask_list

# Import TaskManager from the refactored module
from lm_eval.tasks.manager import TaskManager


__all__ = [
    "TaskManager",
    "get_task_dict",
    "get_task_name_from_config",
    "get_task_name_from_object",
]


def get_task_name_from_config(task_config: dict[str, str]) -> str:
    if "task" in task_config:
        return task_config["task"]
    if "dataset_name" in task_config:
        return "{dataset_path}_{dataset_name}".format(**task_config)
    else:
        return "{dataset_path}".format(**task_config)


def get_task_name_from_object(task_object):
    if hasattr(task_object, "config"):
        return task_object._config["task"]

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def _check_duplicates(task_dict: dict) -> None:
    """helper function solely used in validating get_task_dict output.
    Takes the output of lm_eval.evaluator_utils.get_subtask_list and
    returns a list of all leaf subtasks contained within, and errors if any such leaf subtasks are
    "oversubscribed" to several disjoint groups.
    """
    subtask_names = []
    for key, value in task_dict.items():
        subtask_names.extend(value)

    duplicate_tasks = {
        task_name for task_name in subtask_names if subtask_names.count(task_name) > 1
    }

    # locate the potentially problematic groups that seem to 'compete' for constituent subtasks
    competing_groups = [
        group
        for group in task_dict.keys()
        if len(set(task_dict[group]).intersection(duplicate_tasks)) > 0
    ]

    if len(duplicate_tasks) > 0:
        raise ValueError(
            f"Found 1 or more tasks while trying to call get_task_dict() that were members of more than 1 called group: {list(duplicate_tasks)}. Offending groups: {competing_groups}. Please call groups which overlap their constituent tasks in separate evaluation runs."
        )


def get_task_dict(
    task_name_list: str | list[str | dict | Task],
    task_manager: TaskManager | None = None,
):
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

    task_name_from_string_dict = {}
    task_name_from_config_dict = {}
    task_name_from_object_dict = {}

    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    elif isinstance(task_name_list, list):
        if not all([isinstance(task, (str, dict, Task)) for task in task_name_list]):
            raise TypeError(
                "Expected all list items to be of types 'str', 'dict', or 'Task', but at least one entry did not match."
            )
    else:
        raise TypeError(
            f"Expected a 'str' or 'list' but received {type(task_name_list)}."
        )

    string_task_name_list = [task for task in task_name_list if isinstance(task, str)]
    others_task_name_list = [
        task for task in task_name_list if not isinstance(task, str)
    ]
    if len(string_task_name_list) > 0:
        if task_manager is None:
            task_manager = TaskManager()

        task_name_from_string_dict = task_manager.load_task_or_group(
            string_task_name_list
        )

    for task_element in others_task_name_list:
        if isinstance(task_element, dict):
            task_name_from_config_dict = {
                **task_name_from_config_dict,
                **task_manager.load_config(config=task_element),
            }

        elif isinstance(task_element, Task):
            task_name_from_object_dict = {
                **task_name_from_object_dict,
                get_task_name_from_object(task_element): task_element,
            }

    if not set(task_name_from_string_dict.keys()).isdisjoint(
        set(task_name_from_object_dict.keys())
    ):
        raise ValueError

    final_task_dict = {
        **task_name_from_string_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }

    # behavior can get odd if one tries to invoke several groups that "compete" for the same task.
    # (notably, because one could request several num_fewshot values at once in GroupConfig overrides for the subtask
    # and we'd be unsure which to use and report.)
    # we explicitly check and error in this case.
    _check_duplicates(get_subtask_list(final_task_dict))

    def pretty_print_task(task_name, task_manager, indent: int):
        yaml_path = task_manager.task_index[task_name]["yaml_path"]
        yaml_path = Path(yaml_path)
        lm_eval_tasks_path = Path(__file__).parent
        try:
            display_path = yaml_path.relative_to(lm_eval_tasks_path)
        except ValueError:
            # Path is outside lm_eval/tasks (e.g., from include_path)
            display_path = yaml_path

        pad = "  " * indent
        eval_logger.info(f"{pad}Task: {task_name} ({display_path})")

    # NOTE: Only nicely logs:
    # 1/ group
    #     2/ subgroup
    #         3/ tasks
    # 2/ task
    # layout.
    # TODO: Verify if there are other layouts to nicely display
    eval_logger.info("Selected tasks:")
    for key, value in final_task_dict.items():
        if isinstance(key, ConfigurableGroup):
            eval_logger.info(f"Group: {key.group}")

            if isinstance(value, dict):
                first_key = next(iter(value.keys()))

                if isinstance(first_key, ConfigurableGroup):
                    for subgroup, task_dict in value.items():
                        eval_logger.info(f"  Subgroup: {subgroup.group}")
                        for task_name, configurable_task in task_dict.items():
                            if isinstance(configurable_task, ConfigurableTask):
                                pretty_print_task(task_name, task_manager, indent=2)
                            else:
                                eval_logger.info(f"{task_name}: {configurable_task}")
                else:
                    eval_logger.info(f"{key}: {value}")
            else:
                eval_logger.info(f"{key}: {value}")
        elif isinstance(key, str) and isinstance(value, ConfigurableTask):
            pretty_print_task(key, task_manager, indent=0)
        else:
            eval_logger.info(f"{key}: {value}")

    return final_task_dict
