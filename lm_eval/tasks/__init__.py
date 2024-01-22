import os
import abc
import yaml
import collections

from functools import partial
from typing import List, Union, Dict

from lm_eval import utils
from lm_eval import prompts
from lm_eval.api.task import TaskConfig, Task, ConfigurableTask

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
            tasks = self._get_task_and_group(task_dir)
            ALL_TASKS = {**tasks, **ALL_TASKS}

        return ALL_TASKS

    def all_tasks(self):
        return sorted(list(self.ALL_TASKS.keys()))

    def _name_is_registered(self, name):
        if name in self.ALL_TASKS:
            return True
        return False

    def _name_is_task(self, name):
        if self._name_is_registered(name) and (self.ALL_TASKS[name]["type"] == "task"):
            return True
        return False

    def _config_is_task(self, config):
        if set(config.keys()) <= ["group", "task", "weight_by_size"]:
            return False
        return True

    def _get_yaml_path(self, name):
        assert name in self.ALL_TASKS
        return self.ALL_TASKS[name]["yaml_path"]

    def _get_config(self, name):
        assert name in self.ALL_TASKS
        yaml_path = self._get_yaml_path(name)
        return utils.load_yaml_config(yaml_path)

    def _get_tasklist(self, name):
        assert self._name_is_task(name) == False
        return self.ALL_TASKS[name]["task"]

    def _load_individual_task_or_group(
            self,
            name_or_config: Union[str, dict] = None,
            parent_name: str = None,
            update_config: dict = None
        ) -> ConfigurableTask:

        def load_task(config, task, group=None):
            task_object = ConfigurableTask(config=config)
            if group is not None:
                task_object = (group, task_object)
            return {task: task_object}

        if isinstance(name_or_config, str):
            if update_config is not None:
                name_or_config = {"task": name_or_config, **update_config}
            elif self._name_is_task(name_or_config):
                task_config = self._get_config(name_or_config)
                return load_task(task_config, task=name_or_config, group=parent_name)
            else:
                group_name = name_or_config
                subtask_list = self._get_tasklist(name_or_config)
                if subtask_list == -1:
                    subtask_list = self._get_config(name_or_config)["task"]

        if isinstance(name_or_config, dict):

            if update_config is not None:
                name_or_config={
                    **name_or_config,
                    **update_config,
                }

            if self._config_is_task(name_or_config):
                name = name_or_config["task"]
                if self._name_is_task(name) is False:
                    group_name = name
                    update_config = {k:v for k,v in name_or_config.items() if k is not "task"}
                    subtask_list = self._get_tasklist(name)
                    if subtask_list == -1:
                        subtask_list = self._get_config(name)["task"]
                else:
                    if self._name_is_registered(name):
                        base_task_config = self._get_config(name)
                        task_config={
                                **base_task_config,
                                **name_or_config,
                            }
                    else:
                        task_config = name_or_config
                    return load_task(task_config, task=name, group=parent_name)
            else:
                group_name = name_or_config["group"]
                subtask_list = name_or_config["task"]

        if (self._name_is_registered(group_name) is False) or (self._get_yaml_path(group_name) == -1):
            all_subtasks = {group_name: (parent_name, None)}
        else:
            all_subtasks = {}

        fn = partial(self._load_individual_task_or_group, parent_name=group_name, update_config=update_config)
        all_subtasks = {**all_subtasks, **dict(collections.ChainMap(*map(fn, subtask_list)))}
        return all_subtasks


    def load_task_or_group(self, task_list: Union[str, list] = None) -> dict:

        if isinstance(task_list, str):
            task_list = [task_list]

        all_loaded_tasks = dict(
            collections.ChainMap(
                *map(
                    self._load_individual_task_or_group,
                    task_list
                )
            )
        )
        return all_loaded_tasks

    def _get_task_and_group(self, task_dir: str):
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


# def check_prompt_config(
#     config: Dict[str, str], yaml_path: str = None
# ) -> List[Dict[str, str]]:
#     all_configs = []
#     if "use_prompt" in config:
#         prompt_list = prompts.load_prompt_list(
#             use_prompt=config["use_prompt"],
#             dataset_name=config["dataset_path"],
#             subset_name=config["dataset_name"] if "dataset_name" in config else None,
#             yaml_path=yaml_path,
#         )
#         for idx, prompt_variation in enumerate(prompt_list):
#             all_configs.append(
#                 {
#                     **config,
#                     **{"use_prompt": prompt_variation},
#                     **{
#                         "task": "_".join(
#                             [
#                                 config["task"]
#                                 if "task" in config
#                                 else get_task_name_from_config(config),
#                                 prompt_variation.split("/")[-1]
#                                 if ".yaml" in prompt_variation
#                                 else prompt_variation,
#                             ]
#                         )
#                     },
#                     **{"output_type": "generate_until"},
#                 }
#             )
#     else:
#         all_configs.append(config)
#     return all_configs
