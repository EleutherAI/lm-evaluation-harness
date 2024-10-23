import collections
import inspect
import logging
import os
from functools import partial
from typing import Dict, List, Mapping, Optional, Union

from lm_eval import utils
from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.evaluator_utils import get_subtask_list


GROUP_ONLY_KEYS = list(GroupConfig().to_dict().keys())


class TaskManager:
    """TaskManager indexes all tasks from the default `lm_eval/tasks/`
    and an optional directory if provided.

    """

    def __init__(
        self,
        verbosity="INFO",
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
    ) -> None:
        self.verbosity = verbosity
        self.include_path = include_path
        self.logger = utils.eval_logger
        self.logger.setLevel(getattr(logging, f"{verbosity}"))

        self._task_index = self.initialize_tasks(
            include_path=include_path, include_defaults=include_defaults
        )
        self._all_tasks = sorted(list(self._task_index.keys()))

        self._all_groups = sorted(
            [x for x in self._all_tasks if self._task_index[x]["type"] == "group"]
        )
        self._all_subtasks = sorted(
            [
                x
                for x in self._all_tasks
                if self._task_index[x]["type"] in ["task", "python_task"]
            ]
        )
        self._all_tags = sorted(
            [x for x in self._all_tasks if self._task_index[x]["type"] == "tag"]
        )

        self.task_group_map = collections.defaultdict(list)

    def initialize_tasks(
        self,
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
    ):
        """Creates a dictionary of tasks index.

        :param include_path: Union[str, List] = None
            An additional path to be searched for tasks recursively.
            Can provide more than one such path as a list.
        :param include_defaults: bool = True
            If set to false, default tasks (those in lm_eval/tasks/) are not indexed.
        :return
            Dictionary of task names as key and task metadata
        """
        if include_defaults:
            all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"]
        else:
            all_paths = []
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
    def all_groups(self):
        return self._all_groups

    @property
    def all_subtasks(self):
        return self._all_subtasks

    @property
    def all_tags(self):
        return self._all_tags

    @property
    def task_index(self):
        return self._task_index

    def list_all_tasks(
        self, list_groups=True, list_tags=True, list_subtasks=True
    ) -> str:
        from pytablewriter import MarkdownTableWriter

        def sanitize_path(path):
            # don't print full path if we are within the lm_eval/tasks dir !
            # if we aren't though, provide the full path.
            if "lm_eval/tasks/" in path:
                return "lm_eval/tasks/" + path.split("lm_eval/tasks/")[-1]
            else:
                return path

        group_table = MarkdownTableWriter()
        group_table.headers = ["Group", "Config Location"]
        gt_values = []
        for g in self.all_groups:
            path = self.task_index[g]["yaml_path"]
            if path == -1:
                path = "---"
            else:
                path = sanitize_path(path)
            gt_values.append([g, path])
        group_table.value_matrix = gt_values

        tag_table = MarkdownTableWriter()
        tag_table.headers = ["Tag"]
        tag_table.value_matrix = [[t] for t in self.all_tags]

        subtask_table = MarkdownTableWriter()
        subtask_table.headers = ["Task", "Config Location", "Output Type"]
        st_values = []
        for t in self.all_subtasks:
            path = self.task_index[t]["yaml_path"]

            output_type = ""

            # read the yaml file to determine the output type
            if path != -1:
                config = utils.load_yaml_config(path, mode="simple")
                if "output_type" in config:
                    output_type = config["output_type"]
                elif (
                    "include" in config
                ):  # if no output type, check if there is an include with an output type
                    include_path = path.split("/")[:-1] + config["include"]
                    include_config = utils.load_yaml_config(include_path, mode="simple")
                    if "output_type" in include_config:
                        output_type = include_config["output_type"]

            if path == -1:
                path = "---"
            else:
                path = sanitize_path(path)
            st_values.append([t, path, output_type])
        subtask_table.value_matrix = st_values

        result = "\n"
        if list_groups:
            result += group_table.dumps() + "\n\n"
        if list_tags:
            result += tag_table.dumps() + "\n\n"
        if list_subtasks:
            result += subtask_table.dumps() + "\n\n"
        return result

    def match_tasks(self, task_list):
        return utils.pattern_match(task_list, self.all_tasks)

    def _name_is_registered(self, name) -> bool:
        if name in self.all_tasks:
            return True
        return False

    def _name_is_task(self, name) -> bool:
        if self._name_is_registered(name) and (self.task_index[name]["type"] == "task"):
            return True
        return False

    def _name_is_tag(self, name) -> bool:
        if self._name_is_registered(name) and (self.task_index[name]["type"] == "tag"):
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

    def _class_has_config_in_constructor(self, cls):
        constructor = getattr(cls, "__init__", None)
        return (
            "config" in inspect.signature(constructor).parameters
            if constructor
            else False
        )

    def _load_individual_task_or_group(
        self,
        name_or_config: Optional[Union[str, dict]] = None,
        parent_name: Optional[str] = None,
        update_config: Optional[dict] = None,
    ) -> Mapping:
        def _load_task(config, task):
            if "include" in config:
                config = {
                    **utils.load_yaml_config(
                        yaml_path=None,
                        yaml_config={"include": config.pop("include")},
                        mode="full",
                    ),
                    **config,
                }
            if self._config_is_python_task(config):
                if self._class_has_config_in_constructor(config["class"]):
                    task_object = config["class"](config=config)
                else:
                    task_object = config["class"]()
                if isinstance(task_object, ConfigurableTask):
                    # very scuffed: set task name here. TODO: fixme?
                    task_object.config.task = task
            else:
                task_object = ConfigurableTask(config=config)

            return {task: task_object}

        def _get_group_and_subtask_from_config(config):
            group_name = ConfigurableGroup(config=config)
            subtask_list = []
            for task in group_name.config["task"]:
                if isinstance(task, str) and self._name_is_tag(task):
                    subtask_list.extend(self._get_tasklist(task))
                else:
                    subtask_list.append(task)
            return group_name, subtask_list

        def _process_group_config(config, update_config=None):
            if update_config is not None:
                config = {**config, **update_config}
            _update_config = {
                k: v for k, v in config.items() if k not in GROUP_ONLY_KEYS
            }
            if not bool(_update_config):
                _update_config = None

            group_config = {k: v for k, v in config.items() if k in GROUP_ONLY_KEYS}
            return group_config, _update_config

        if isinstance(name_or_config, str):
            if update_config is not None:
                # Process name_or_config as a dict instead
                name_or_config = {"task": name_or_config, **update_config}
            elif self._name_is_task(name_or_config) or self._name_is_python_task(
                name_or_config
            ):
                task_config = self._get_config(name_or_config)
                return _load_task(task_config, task=name_or_config)
            else:
                subtask_list = self._get_tasklist(name_or_config)
                if subtask_list == -1:
                    group_config = self._get_config(name_or_config)
                    group_config, update_config = _process_group_config(group_config)
                    group_name, subtask_list = _get_group_and_subtask_from_config(
                        group_config
                    )
                else:
                    if self._name_is_tag(name_or_config):
                        fn = partial(
                            self._load_individual_task_or_group,
                            update_config=name_or_config
                            if isinstance(name_or_config, dict)
                            else None,
                        )
                        return dict(
                            collections.ChainMap(*map(fn, reversed(subtask_list)))
                        )
                    else:
                        group_name = ConfigurableGroup(
                            config={"group": name_or_config, "task": subtask_list}
                        )

        if isinstance(name_or_config, dict):
            if self._config_is_task(name_or_config):
                name = name_or_config.pop("task")
                if update_config is not None:
                    name_or_config = {**name_or_config, **update_config}
                # If the name is registered as a group
                if self._name_is_group(name):
                    group_config = self._get_config(name)

                    group_config, update_config = _process_group_config(
                        group_config, name_or_config
                    )
                    group_name, subtask_list = _get_group_and_subtask_from_config(
                        group_config
                    )
                elif self._name_is_tag(name):
                    subtask_list = self._get_tasklist(name)
                    fn = partial(
                        self._load_individual_task_or_group,
                        update_config=name_or_config,
                    )
                    return dict(collections.ChainMap(*map(fn, reversed(subtask_list))))
                else:
                    if self._name_is_registered(name):
                        base_task_config = self._get_config(name)

                        # Check if this is a duplicate.
                        if parent_name is not None:
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
                    return _load_task(task_config, task=name)
            else:
                group_config, update_config = _process_group_config(name_or_config)
                group_name, subtask_list = _get_group_and_subtask_from_config(
                    group_config
                )

        fn = partial(
            self._load_individual_task_or_group,
            parent_name=group_name,
            update_config=update_config,
        )
        return {
            group_name: dict(collections.ChainMap(*map(fn, reversed(subtask_list))))
        }

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
        - `type`, that can be either `task`, `python_task`, `group` or `tags`.
            `task` refer to regular task configs, `python_task` are special
            yaml files that only consists of `task` and `class` parameters.
            `group` are group configs. `tags` are labels that can be assigned
            to tasks to assist in sorting and calling tasks of certain themes.
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

        def _populate_tags_and_groups(config, task, tasks_and_groups, print_info):
            # TODO: remove group in next release
            if "tag" in config:
                attr_list = config["tag"]
                if isinstance(attr_list, str):
                    attr_list = [attr_list]

                for tag in attr_list:
                    if tag not in tasks_and_groups:
                        tasks_and_groups[tag] = {
                            "type": "tag",
                            "task": [task],
                            "yaml_path": -1,
                        }
                    elif tasks_and_groups[tag]["type"] != "tag":
                        self.logger.info(
                            f"The tag '{tag}' is already registered as a group, this tag will not be registered. "
                            "This may affect tasks you want to call."
                        )
                        break
                    else:
                        tasks_and_groups[tag]["task"].append(task)

        # TODO: remove group in next release
        print_info = True
        ignore_dirs = [
            "__pycache__",
            ".ipynb_checkpoints",
        ]
        tasks_and_groups = collections.defaultdict()
        for root, dirs, file_list in os.walk(task_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    if self._config_is_python_task(config):
                        # This is a python class config
                        task = config["task"]
                        tasks_and_groups[task] = {
                            "type": "python_task",
                            "yaml_path": yaml_path,
                        }
                        _populate_tags_and_groups(
                            config, task, tasks_and_groups, print_info
                        )
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
                        _populate_tags_and_groups(
                            config, task, tasks_and_groups, print_info
                        )
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


def _check_duplicates(task_dict: dict) -> List[str]:
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

    return final_task_dict
