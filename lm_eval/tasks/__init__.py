import collections
import logging
import os
from functools import partial
from typing import Dict, List, Mapping, Optional, Union

from lm_eval import utils
from lm_eval.api.task import ConfigurableTask, Task


class TaskManager:
    """TaskManager indexes all tasks from the default `lm_eval/tasks/`
    and an optional directory if provided.

    """

    def __init__(self, verbosity="INFO", include_path: Optional[str] = None) -> None:
        self.verbosity = verbosity
        self.include_path = include_path
        self.logger = utils.eval_logger
        self.logger.setLevel(getattr(logging, f"{verbosity}"))

        self._task_index = self.initialize_tasks(include_path=include_path)
        self._all_tasks = sorted(list(self._task_index.keys()))

        self.task_group_map = collections.defaultdict(list)

    def initialize_tasks(self, include_path: Optional[str] = None):
        """Creates a dictionary of tasks index.

        :param include_path: str = None
            An additional path to be searched for tasks

        :return
            Dictionary of task names as key and task metadata
        """
        all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"]
        if include_path is not None:
            if isinstance(include_path, str):
                include_path = [include_path]
            all_paths.extend(include_path)

        task_index = {}
        for task_dir in all_paths:
            tasks = self._get_task_and_group(task_dir)
            task_index = {**tasks, **task_index}

        return task_index

    @property
    def all_tasks(self):
        return self._all_tasks

    @property
    def task_index(self):
        return self._task_index

    def match_tasks(self, task_list):
        return utils.pattern_match(task_list, self.all_tasks)

    def _name_is_registered(self, name) -> bool:
        if name in self.all_tasks:
            return True
        return False

    def _name_is_task(self, name) -> bool:
        if self._name_is_registered(name) and ("task" in self.task_index[name]["type"]):
            return True
        return False

    def _name_is_group(self, name) -> bool:
        if self._name_is_registered(name) and (
            self.task_index[name]["type"] == "group"
        ):
            return True
        return False

    def _name_is_python_task(self, name):
        if self._name_is_registered(name) and (
            self.task_index[name]["type"] == "python_task"
        ):
            return True
        return False

    def _config_is_task(self, config) -> bool:
        if ("task" in config) and isinstance(config["task"], str):
            return True
        return False

    def _config_is_group(self, config) -> bool:
        if ("task" in config) and isinstance(config["task"], list):
            return True
        return False

    def _config_is_python_task(self, config) -> bool:
        if "class" in config:
            return True
        return False

    def _get_yaml_path(self, name):
        if name not in self.task_index:
            raise ValueError
        return self.task_index[name]["yaml_path"]

    def _get_config(self, name):
        if name not in self.task_index:
            raise ValueError
        yaml_path = self._get_yaml_path(name)
        if yaml_path == -1:
            return {}
        else:
            return utils.load_yaml_config(yaml_path, mode="full")

    def _get_tasklist(self, name):
        if self._name_is_task(name):
            raise ValueError
        return self.task_index[name]["task"]

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
        name_or_config: Optional[Union[str, dict]] = None,
        parent_name: Optional[str] = None,
        update_config: Optional[dict] = None,
        yaml_path: Optional[str] = None,
    ) -> Mapping:
        def load_task(config, task, group=None, yaml_path=None):
            if "include" in config:
                if yaml_path is None:
                    raise ValueError
                config.update(
                    utils.load_yaml_config(
                        yaml_path,
                        yaml_config={"include": config.pop("include")},
                        mode="full",
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
                    group_config = self._get_config(name_or_config)
                    subtask_list = group_config["task"]

                # This checks if we're at the root.
                if parent_name is None:
                    group_config = self._get_config(name_or_config)
                    if set(group_config.keys()) > {"task", "group"}:
                        update_config = {
                            k: v
                            for k, v in group_config.items()
                            if k not in ["task", "group"]
                        }
                    yaml_path = self._get_yaml_path(group_name)

                    if (update_config is not None) and ("group_alias" in update_config):
                        group_name = update_config["group_alias"]
                        update_config.pop("group_alias")

        if isinstance(name_or_config, dict):
            if update_config is not None:
                name_or_config = {
                    **name_or_config,
                    **update_config,
                }

            if self._config_is_task(name_or_config):
                name = name_or_config["task"]
                # If the name is registered as a group
                # if self._name_is_task(name) is False:
                if self._name_is_group(name):
                    group_name = name
                    update_config = {
                        k: v for k, v in name_or_config.items() if k != "task"
                    }
                    subtask_list = self._get_tasklist(name)
                    if subtask_list == -1:
                        subtask_list = self._get_config(name)["task"]
                else:
                    if self._name_is_registered(name):
                        base_task_config = self._get_config(name)

                        # Check if this is a duplicate.
                        if parent_name is not None:
                            name_or_config["group"] = parent_name
                            num_duplicate = len(
                                list(
                                    filter(
                                        lambda x: x.startswith(name),
                                        self.task_group_map[parent_name],
                                    )
                                )
                            )
                            if num_duplicate > 0:
                                name = f"{name}-{num_duplicate}"
                            self.task_group_map[parent_name].append(name)

                        task_config = {
                            **base_task_config,
                            **name_or_config,
                        }
                    else:
                        task_config = name_or_config
                    return load_task(
                        task_config, task=name, group=parent_name, yaml_path=yaml_path
                    )
            else:
                group_name = name_or_config["group"]
                subtask_list = name_or_config["task"]
                if set(name_or_config.keys()) > {"task", "group"}:
                    update_config = {
                        k: v
                        for k, v in name_or_config.items()
                        if k not in ["task", "group"]
                    }

        all_subtasks = {}
        if parent_name is not None:
            all_subtasks = {group_name: (parent_name, None)}

        fn = partial(
            self._load_individual_task_or_group,
            parent_name=group_name,
            update_config=update_config,
            yaml_path=yaml_path,
        )
        all_subtasks = {
            **all_subtasks,
            **dict(collections.ChainMap(*map(fn, subtask_list))),
        }
        return all_subtasks

    def load_task_or_group(self, task_list: Optional[Union[str, list]] = None) -> dict:
        """Loads a dictionary of task objects from a list

        :param task_list: Union[str, list] = None
            Single string or list of string of task names to be loaded

        :return
            Dictionary of task objects
        """
        if isinstance(task_list, str):
            task_list = [task_list]

        all_loaded_tasks = dict(
            collections.ChainMap(*map(self._load_individual_task_or_group, task_list))
        )
        return all_loaded_tasks

    def load_config(self, config: Dict):
        return self._load_individual_task_or_group(config)

    def _get_task_and_group(self, task_dir: str):
        """Creates a dictionary of tasks index with the following metadata,
        - `type`, that can be either `task`, `python_task`, or `group`.
            `task` refer to regular task configs, `python_task` are special
            yaml files that only consists of `task` and `class` parameters.
            `group` are group configs.
        - `yaml_path`, path to the yaml file. If the entry is a `group` that
            was configured through a task config, the yaml_path will be -1
            and all subtasks will be listed in `task` (see below)
        - `task`, reserved for entries with `type` as `group`. This will list
            all subtasks. When a group config is created (as opposed to task
            config having `group` parameter set), this will be set to -1 to
            avoid recursive indexing. The whole list of subtasks will be loaded
            at evaluation.

        :param task_dir: str
            A directory to check for tasks

        :return
            Dictionary of task names as key and task metadata
        """
        tasks_and_groups = collections.defaultdict()
        for root, _, file_list in os.walk(task_dir):
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    config = utils.load_yaml_config(yaml_path, mode="simple")
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
                            "task": -1,  # This signals that
                            # we don't need to know
                            # the task list for indexing
                            # as it can be loaded
                            # when called.
                            "yaml_path": yaml_path,
                        }

                        # # Registered the level 1 tasks from a group config
                        # for config in config["task"]:
                        #     if isinstance(config, dict) and self._config_is_task(config):
                        #         task = config["task"]
                        #         tasks_and_groups[task] = {
                        #             "type": "task",
                        #             "yaml_path": yaml_path,
                        #             }

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


def get_task_name_from_config(task_config: Dict[str, str]) -> str:
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


def get_task_dict(
    task_name_list: Union[str, List[Union[str, Dict, Task]]],
    task_manager: Optional[TaskManager] = None,
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
    others_task_name_list = [task for task in task_name_list if ~isinstance(task, str)]
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

    return {
        **task_name_from_string_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }
