import os
import abc
import yaml
import collections
from typing import List, Union, Dict

from lm_eval import utils
from lm_eval import prompts
from lm_eval.api.task import TaskConfig, Task, ConfigurableTask
from lm_eval.api.registry import (
    register_task,
    register_group,
    TASK_REGISTRY,
    GROUP_REGISTRY,
)

import logging

# import python tasks
from .squadv2.task import SQuAD2
from .scrolls.task import (
    QuALITY,
    NarrativeQA,
    ContractNLI,
    GovReport,
    SummScreenFD,
    QMSum,
)

eval_logger = utils.eval_logger


def is_group(task):
    if list(task.keys()) == ["group", "task"]:
        return True
    return False

class TaskManager(abc.ABC):

    def __init__(
        self, 
        verbosity="INFO",
        include_path=None
        ) -> None:

        self.verbosity = verbosity
        self.include_path = include_path
        self.logger = eval_logger.setLevel(getattr(logging, f"{verbosity}"))

        self.ALL_TASKS = self.initialize_tasks(
            include_path=include_path
            )

    def initialize_tasks(self, include_path=None):
        
        all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"]
        if include_path is not None:
            if isinstance(include_path, str):
                include_path = [include_path]
            all_paths.extend(include_path)

        ALL_TASKS = {}
        for task_dir in all_paths:
            tasks = get_task_and_group(task_dir)
            ALL_TASKS = {**tasks, **ALL_TASKS}

        return ALL_TASKS

    def all_tasks(self):
        return sorted(list(self.ALL_TASKS.keys()))

    def _load_individual_task_or_group(self, task_name_or_config: Union[str, dict] = None) -> ConfigurableTask:

        print("Loading", task_name_or_config)
        if isinstance(task_name_or_config, str):
            task_info = self.ALL_TASKS[task_name_or_config]
            yaml_path = task_info["yaml_path"]
            task_type = task_info["type"]
            subtask_list = task_info["task"] if "task" in task_info else -1
            if task_type == "task":
                task_config = utils.load_yaml_config(yaml_path)
                return ConfigurableTask(config=task_config)
            else:
                if subtask_list == -1:
                    task_config = utils.load_yaml_config(yaml_path)
                    group_name = task_config["group"]
                    subtask_list = task_config["task"]
                else:
                    group_name = task_name_or_config

                all_subtasks = {}
                for task_or_config in subtask_list:
                    if isinstance(task_or_config, str):
                        all_subtasks[task_or_config] = (group_name, None)
                        task_object = self._load_individual_task_or_group(task_name_or_config=task_or_config)
                    elif isinstance(task_or_config, dict):
                        if "group" in task_or_config:
                            all_subtasks[task_or_config["group"]] = (group_name, None)
                        elif "task" in task_or_config:
                            all_subtasks[task_or_config["task"]] = (group_name, None)
                        task_object = self._load_individual_task_or_group(task_name_or_config=task_or_config)

                    if isinstance(task_object, dict):
                        all_subtasks = {**task_object, **all_subtasks}
                    else:
                        task_name = task_object._config["task"]
                        all_subtasks[task_name] = (group_name, task_object)
                        # if group_name is not None:
                        #     all_subtasks[task_name] = (group_name, task_object)
                        # else:
                        #     all_subtasks[task_name] = task_object
                return all_subtasks
        elif isinstance(task_name_or_config, dict):
            if is_group(task_name_or_config):
                group_name = task_name_or_config["group"]
                subtask_list = task_name_or_config["task"]
                all_subtasks = {}
                for task_or_config in subtask_list:
                    if isinstance(task_or_config, str):
                        task_object = self._load_individual_task_or_group(task_name_or_config=task_or_config)
                        task_name = task_or_config
                    elif isinstance(task_or_config, dict):
                        task_object = self._load_individual_task_or_group(task_name_or_config=task_or_config)

                    if isinstance(task_object, dict):
                        all_subtasks = {**task_object, **all_subtasks}
                    else:
                        task_name = task_object._config["task"]
                        all_subtasks[task_name] = (group_name, task_object)
                return all_subtasks
            else:
                task_type = "task"
                task_name = task_name_or_config["task"]
                base_task_info = self.ALL_TASKS[task_name]
                base_yaml_path = base_task_info["yaml_path"]
                base_task_config = utils.load_yaml_config(base_yaml_path)

                return ConfigurableTask(
                    config={
                        **base_task_config,
                        **task_name_or_config,
                    }
                )

    def load_task_or_group(self, task_list: Union[str, list] = None) -> dict:

        if isinstance(task_list, str):
            task_list = [task_list]

        all_loaded_tasks = {}
        for task in task_list:
            task_object = self._load_individual_task_or_group(
                task_name_or_config=task,
            )
            if isinstance(task, str):
                task_name = task
            elif isinstance(task, dict):
                task_name = task["task"]

            if isinstance(task_object, dict):
                all_loaded_tasks = {**task_object, **all_loaded_tasks}
            else:
                all_loaded_tasks[task_name] = task_object
        
        return all_loaded_tasks


def register_configurable_task(config: Dict[str, str]) -> int:
    SubClass = type(
        config["task"] + "ConfigurableTask",
        (ConfigurableTask,),
        {"CONFIG": TaskConfig(**config)},
    )

    if "task" in config:
        task_name = "{}".format(config["task"])
        register_task(task_name)(SubClass)

    if "group" in config:
        if config["group"] == config["task"]:
            raise ValueError("task and group name cannot be the same")
        elif type(config["group"]) == str:
            group_name = [config["group"]]
        else:
            group_name = config["group"]

        for group in group_name:
            register_group(group)(SubClass)

    return 0


def register_configurable_group(config: Dict[str, str], yaml_path: str = None) -> int:
    group = config["group"]

    task_config_list = []
    group_config_list = []
    registered_task_or_group_list = []
    for task in config["task"]:
        if isinstance(task, str):
            registered_task_or_group_list.append(task)
        elif is_group(task):
            group_config_list.append(task)
        else:
            task_config_list.append(task)

    for task_config in task_config_list:
        base_config = {}
        task_name_config = {}
        if "task" in task_config:
            task_name = task_config["task"]
            if task_name in TASK_REGISTRY:
                task_obj = get_task_dict(task_name)[task_name]
                if type(task_obj) == tuple:
                    _, task_obj = task_obj

                if task_obj is not None:
                    base_config = task_obj._config.to_dict(keep_callable=True)
                    task_name_config["task"] = f"{group}_{task_name}"

        task_config = utils.load_yaml_config(yaml_path, task_config)
        var_configs = check_prompt_config(
            {
                **base_config,
                **task_config,
                **{"group": group},
                **task_name_config,
            },
            yaml_path=os.path.dirname(yaml_path),
        )
        for config in var_configs:
            register_configurable_task(config)

    for group_config in group_config_list:
        sub_group = group_config["group"]
        register_configurable_group(group_config, yaml_path)
        if group in GROUP_REGISTRY:
            GROUP_REGISTRY[group].append(sub_group)
        else:
            GROUP_REGISTRY[group] = [sub_group]
            self.ALL_TASKS.add(group)

    task_names = utils.pattern_match(registered_task_or_group_list, self.ALL_TASKS)
    for task in task_names:
        if (task in TASK_REGISTRY) or (task in GROUP_REGISTRY):
            if group in GROUP_REGISTRY:
                GROUP_REGISTRY[group].append(task)
            else:
                GROUP_REGISTRY[group] = [task]
                self.ALL_TASKS.add(group)

    return 0


def check_prompt_config(
    config: Dict[str, str], yaml_path: str = None
) -> List[Dict[str, str]]:
    all_configs = []
    if "use_prompt" in config:
        prompt_list = prompts.load_prompt_list(
            use_prompt=config["use_prompt"],
            dataset_name=config["dataset_path"],
            subset_name=config["dataset_name"] if "dataset_name" in config else None,
            yaml_path=yaml_path,
        )
        for idx, prompt_variation in enumerate(prompt_list):
            all_configs.append(
                {
                    **config,
                    **{"use_prompt": prompt_variation},
                    **{
                        "task": "_".join(
                            [
                                config["task"]
                                if "task" in config
                                else get_task_name_from_config(config),
                                prompt_variation.split("/")[-1]
                                if ".yaml" in prompt_variation
                                else prompt_variation,
                            ]
                        )
                    },
                    **{"output_type": "generate_until"},
                }
            )
    else:
        all_configs.append(config)
    return all_configs


def get_task_name_from_config(task_config: Dict[str, str]) -> str:
    if "dataset_name" in task_config:
        return "{dataset_path}_{dataset_name}".format(**task_config)
    else:
        return "{dataset_path}".format(**task_config)


def include_task_folder(task_dir: str, register_task: bool = True, task_name: str = None) -> None:
    """
    Calling this function
    """

    # Track whether any tasks failed during loading
    import_fail = False
    for root, subdirs, file_list in os.walk(task_dir):
        # if (subdirs == [] or subdirs == ["__pycache__"]) and (len(file_list) > 0):
        for f in file_list:
            if f.endswith(".yaml"):
                yaml_path = os.path.join(root, f)
                try:
                    config = utils.load_yaml_config(yaml_path)

                    if "task" not in config:
                        continue

                    all_configs = check_prompt_config(
                        config, yaml_path=os.path.dirname(yaml_path)
                    )
                    for config in all_configs:
                        if register_task:
                            if type(config["task"]) == str:
                                register_configurable_task(config)
                        else:
                            if type(config["task"]) == list:
                                register_configurable_group(config, yaml_path)

                # Log this silently and show it only when
                # the user defines the appropriate verbosity.
                except (ImportError, ModuleNotFoundError) as e:
                    import_fail = True
                    eval_logger.debug(
                        f"{yaml_path}: {e}. Config will not be added to registry."
                    )
                except Exception as error:
                    import traceback

                    eval_logger.warning(
                        "Unexpected error loading config in\n"
                        f"                                 {yaml_path}\n"
                        "                                 Config will not be added to registry\n"
                        f"                                 Error: {error}\n"
                        f"                                 Traceback: {traceback.format_exc()}"
                    )

    if import_fail:
        eval_logger.warning(
          "Some tasks could not be loaded due to missing dependencies."
          " Run with `--verbosity DEBUG` for full details."
          )
    return 0


def get_task_and_group(task_dir: str):
    tasks_and_groups = collections.defaultdict()
    for root, _, file_list in os.walk(task_dir):
        for f in file_list:
            if f.endswith(".yaml"):
                yaml_path = os.path.join(root, f)
                config = utils.simple_load_yaml_config(yaml_path)
                if list(config.keys()) == ["group", "task"]:
                    # This is a group config
                    tasks_and_groups[config["group"]] = {
                        "type": "group",
                        "task": -1, # This signals that
                                    # we don't need to know
                                    # the task list for indexing
                                    # as it can be loaded
                                    # when called.
                        "yaml_path": yaml_path,
                    }
                else:
                    # This is a task config
                    task = config["task"]
                    tasks_and_groups[task] = {
                        "type": "task",
                        "yaml_path": yaml_path,
                        }

                    if "group" in config:
                        groups = config["group"]
                        if isinstance(config["group"], str):
                            groups = [groups]

                        for group in groups:
                            if group not in tasks_and_groups:
                                tasks_and_groups[group] = {
                                    "type": "group",
                                    "task": [task],
                                    "yaml_path": -1,
                                }
                            else:
                                tasks_and_groups[group]["task"].append(task)

    return tasks_and_groups



def get_task(task_name, config):
    try:
        return TASK_REGISTRY[task_name](config=config)
    except KeyError:
        eval_logger.info("Available tasks:")
        eval_logger.info(list(TASK_REGISTRY) + list(GROUP_REGISTRY))
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


# TODO: pass num_fewshot and other cmdline overrides in a better way
def get_task_dict(task_name_list: List[Union[str, Dict, Task]], **kwargs):
    config = {**kwargs}

    task_name_from_registry_dict = {}
    task_name_from_config_dict = {}
    task_name_from_object_dict = {}

    if type(task_name_list) != list:
        task_name_list = [task_name_list]

    for task_element in task_name_list:
        if isinstance(task_element, str):
            if task_element in GROUP_REGISTRY:
                group_name = task_element
                for task_name in GROUP_REGISTRY[task_element]:
                    if task_name not in task_name_from_registry_dict:
                        task_obj = get_task_dict(task_name)
                        if task_name in task_obj.keys():
                            task_dict = {
                                task_name: (group_name, task_obj[task_name]),
                            }
                        else:
                            task_dict = {
                                task_name: (group_name, None),
                                **task_obj,
                            }

                        task_name_from_registry_dict = {
                            **task_name_from_registry_dict,
                            **task_dict,
                        }
            else:
                task_name = task_element
                if task_name not in task_name_from_registry_dict:
                    task_name_from_registry_dict = {
                        **task_name_from_registry_dict,
                        task_name: get_task(task_name=task_element, config=config),
                    }

        elif isinstance(task_element, dict):
            task_element.update(config)
            task_name_from_config_dict = {
                **task_name_from_config_dict,
                get_task_name_from_config(task_element): ConfigurableTask(
                    config=task_element
                ),
            }

        elif isinstance(task_element, Task):
            task_name_from_object_dict = {
                **task_name_from_object_dict,
                get_task_name_from_object(task_element): task_element,
            }

    assert set(task_name_from_registry_dict.keys()).isdisjoint(
        set(task_name_from_object_dict.keys())
    )
    return {
        **task_name_from_registry_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }
