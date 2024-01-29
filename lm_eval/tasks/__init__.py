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


class TaskManager(abc.ABC):

    def __init__(
        self,
        verbosity="INFO",
        include_path=None
        ) -> None:

        self.verbosity = verbosity
        self.include_path = include_path
        self.logger = utils.eval_logger
        self.logger.setLevel(getattr(logging, f"{verbosity}"))

        self.ALL_TASKS = self.initialize_tasks(
            include_path=include_path
            )

        self.task_group_map = collections.defaultdict(list)

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
        if "task" in self.ALL_TASKS[name]["type"]:
            return True
        return False

    def _name_is_python_task(self, name):
        if self._name_is_registered(name) and (self.ALL_TASKS[name]["type"] == "python_task"):
            return True
        return False

    def _config_is_task(self, config):
        if ("task" in config) and isinstance(config["task"], str):
            return True
        return False

    def _config_is_group(self, config):
        if ("task" in config) and isinstance(config["task"], list):
            return True
        return False

    def _config_is_python_task(self, config):
        if "class" in config:
            return True
        return False

    def _get_yaml_path(self, name):
        assert name in self.ALL_TASKS
        return self.ALL_TASKS[name]["yaml_path"]

    def _get_config(self, name):
        assert name in self.ALL_TASKS
        yaml_path = self._get_yaml_path(name)
        return utils.load_yaml_config("full", yaml_path)

    def _get_tasklist(self, name):
        assert self._name_is_task(name) == False
        return self.ALL_TASKS[name]["task"]

    def _process_alias(self, config, group=None):
        # If the group is not the same as the original 
        # group which the group alias was intended for,
        # Set the group_alias to None instead.
        if ("group_alias" in config) and ("group" in config) and group is not None:
            if config["group"] != group:
                config["group_alias"] = None
        return config

    def _load_individual_task_or_group(
            self,
            name_or_config: Union[str, dict] = None,
            parent_name: str = None,
            update_config: dict = None,
            yaml_path: str = None,
        ) -> ConfigurableTask:

        def load_task(config, task, group=None, yaml_path=None):
            if "include" in config:
                assert yaml_path is not None
                config.update(
                    utils.load_yaml_config(
                        "full",
                        yaml_path,
                        yaml_config={"include": config.pop("include")}
                    )
                )
            if self._config_is_python_task(config):
                task_object = config["class"]()
            else:
                config = self._process_alias(config, group=group)
                task_object = ConfigurableTask(config=config)
            if group is not None:
                task_object = (group, task_object)
            return {task: task_object}

        if isinstance(name_or_config, str):
            if update_config is not None:
                # Process name_or_config as a dict instead
                name_or_config = {"task": name_or_config, **update_config}
            elif self._name_is_task(name_or_config):
                task_config = self._get_config(name_or_config)
                return load_task(task_config, task=name_or_config, group=parent_name)
            else:
                group_name = name_or_config
                subtask_list = self._get_tasklist(name_or_config)
                if subtask_list == -1:
                    subtask_list = self._get_config(name_or_config)["task"]

                # This checks if we're at the root.
                if parent_name is None:
                    yaml_path = self._get_yaml_path(group_name)

        if isinstance(name_or_config, dict):

            if update_config is not None:
                name_or_config={
                    **name_or_config,
                    **update_config,
                }

            if self._config_is_task(name_or_config):
                name = name_or_config["task"]
                # If the name is registered as a group
                if self._name_is_registered(name):
                    if self._name_is_task(name) is False:
                        group_name = name
                        update_config = {k:v for k,v in name_or_config.items() if k != "task"}
                        subtask_list = self._get_tasklist(name)
                        if subtask_list == -1:
                            subtask_list = self._get_config(name)["task"]
                    else:
                        base_task_config = self._get_config(name)

                        # Check if this is a duplicate.
                        if parent_name is not None:
                            num_duplicate = len(list(filter(lambda x: x.startswith(name), self.task_group_map[parent_name])))
                            if num_duplicate > 0:
                                name = f"{name}-{num_duplicate}"
                            self.task_group_map[parent_name].append(name)

                        task_config={
                                **base_task_config,
                                **name_or_config,
                            }
                else:
                    task_config = name_or_config
                return load_task(task_config, task=name, group=parent_name, yaml_path=yaml_path)
            else:
                group_name = name_or_config["group"]
                subtask_list = name_or_config["task"]

        all_subtasks = {}
        if (parent_name is not None) and ((self._name_is_registered(group_name) is False) or (self._get_yaml_path(group_name) == -1)):
            all_subtasks = {group_name: (parent_name, None)}

        fn = partial(self._load_individual_task_or_group, parent_name=group_name, update_config=update_config, yaml_path=yaml_path)
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
                    config = utils.load_yaml_config("simple", yaml_path)
                    if self._config_is_python_task(config):
                        # This is a python class config
                        tasks_and_groups[config["task"]] = {
                            "type": "python_task",
                            "yaml_path": yaml_path,
                        }
                    elif self._config_is_group(config):
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

                        # for config in config["task"]:
                        #     if isinstance(config, dict) and self._config_is_task(config):
                        #         config["task"]

                    elif self._config_is_task(config):
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
                    else:
                        self.logger.debug(f"File {f} in {root} could not be loaded")

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
