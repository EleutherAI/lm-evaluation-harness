import collections
import functools
import importlib.util
import inspect
import logging
import sys
from functools import partial
from glob import iglob
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Union

import yaml
from yaml import YAMLError

from lm_eval import utils
from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.evaluator_utils import get_subtask_list


GROUP_ONLY_KEYS = list(GroupConfig().to_dict().keys())
_CONFIG_CACHE: dict[tuple[Path, str], dict] = {}


eval_logger = logging.getLogger(__name__)
_Base = yaml.CLoader if getattr(yaml, "__with_libyaml__", False) else yaml.FullLoader


@functools.lru_cache(maxsize=128)  # ← reuse per (directory, simple) pair
def _make_loader(yaml_dir: Path, simple: bool = False) -> type[yaml.Loader]:
    """
    Return a custom YAML Loader class bound to *yaml_dir*.

    yaml_dir
        Directory that holds the YAML file being parsed.
        We capture it so that !function look-ups can resolve relative
        Python files like  my_utils.some_fn  ➜  yaml_dir / "my_utils.py".
    simple
        If True we ignore !function completely (used by `mode="simple"`).
    """

    class Loader(_Base):
        """Dynamically-generated loader that knows its base directory."""

        # no extra state needed; the constructor stays the same

    # Register (or stub) the !function constructor **for this Loader only**
    if simple:
        yaml.add_constructor("!function", lambda *_: None, Loader=Loader)
    else:
        yaml.add_constructor(
            "!function",
            # capture yaml_dir once so the lambda is fast and pickle-able
            lambda ld, node, _dir=yaml_dir: _import_function(
                ld.construct_scalar(node),
                base_path=_dir,
            ),
            Loader=Loader,
        )

    return Loader


@functools.lru_cache(maxsize=1000)  # ← cache module objects
def _import_function(qualname: str, *, base_path: Path) -> Callable:
    mod_path, _, func_name = qualname.rpartition(".")
    if not mod_path:
        raise ValueError(f"{qualname!r} has no module part")
    file_path = base_path / f"{mod_path.replace('.', '/')}.py"
    module_name = f"_yaml_dynamic.{hash(file_path)}_{file_path.stem}"
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[module_name] = mod
    return getattr(mod, func_name)


def ignore_constructor(loader: yaml.Loader, node: yaml.Node) -> None:
    return None


@functools.lru_cache(maxsize=1000)  #
def _parse_yaml_file(path: Path, mode: str) -> dict:
    loader_cls = _make_loader(path.parent, simple=(mode == "simple"))
    with path.open("rb") as fh:
        return yaml.load(fh, Loader=loader_cls)


def load_yaml_config(
    yaml_path: Union[Path, str, None] = None,
    yaml_config: dict | None = None,
    yaml_dir: Path | None = None,
    mode: str = "full",
    *,
    _seen: set[tuple[Path, str]] | None = None,
    resolve_includes: bool = True,
) -> dict:
    """
    Parse a YAML config with optional include handling.

    Parameters
    ----------
    yaml_path
        Path to the main YAML file.  Needed unless *yaml_config* is
        supplied directly (e.g. by tests).
    yaml_config
        Pre-parsed dict to use instead of reading *yaml_path*.
    yaml_dir
        Base directory for resolving relative include paths.  Defaults
        to `yaml_path.parent`.
    mode
        "full"  – honour  !function  tags
        "simple" – ignore !function  (faster).
    _seen
        **Internal** recursion set: tuples of (absolute-path, mode).
        Prevents include cycles such as  A → B → A.
    """
    if yaml_config is None and yaml_path is None:
        raise ValueError("load_yaml_config needs either yaml_path or yaml_config")

    # ------------------------------------------------------------------ cycle guard
    if _seen is None:
        _seen = set()
    if yaml_path is not None:
        yaml_path = Path(yaml_path).expanduser().resolve()

        # ---------- fast-path: return memoised, already-resolved cfg ----------
        cache_key = (yaml_path, mode)
        if yaml_config is None and resolve_includes and cache_key in _CONFIG_CACHE:
            return _CONFIG_CACHE[cache_key]

        key = (yaml_path.resolve(), mode)
        if key in _seen:
            raise ValueError(f"Include cycle detected at {yaml_path}")
        _seen.add(key)

    # ------------------------------------------------------------------ load / parse
    if yaml_config is None:  # ordinary path-based load
        yaml_config = _parse_yaml_file(yaml_path, mode)

    if yaml_dir is None and yaml_path is not None:
        yaml_dir = yaml_path.parent
    assert yaml_dir is not None, "yaml_dir must be set by caller or deduced from path"

    # ------------------------------------------------------------------ handle include
    include = yaml_config.pop("include", None)
    if not include and not resolve_includes:
        return yaml_config

    include_paths = include if isinstance(include, list) else [include]
    final_cfg: dict = {}

    for inc in reversed(include_paths):
        if inc is None:  # guard against explicit nulls
            continue
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = (yaml_dir / inc_path).resolve()
        included = load_yaml_config(
            yaml_path=inc_path,
            mode=mode,
            yaml_dir=inc_path.parent,
            _seen=_seen,  # <-- pass set downward
        )
        final_cfg.update(included)

    final_cfg.update(yaml_config)  # local keys win
    # -------- memoise after *all* includes are merged ----------
    if yaml_config is None and resolve_includes:
        _CONFIG_CACHE[cache_key] = final_cfg

    return final_cfg


def iter_yaml_files(root: Path) -> Generator[Path, Any, None]:
    # '**/*.yaml' is handled internally by os.scandir.
    for path in iglob("**/*.yaml", root_dir=root, recursive=True):
        # quick ignore check
        if "/__pycache__/" in path or "/.ipynb_checkpoints/" in path:
            continue
        yield root / path


class TaskManager:
    """TaskManager indexes all tasks from the default `lm_eval/tasks/`
    and an optional directory if provided.

    """

    def __init__(
        self,
        verbosity: Optional[str] = None,
        include_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        include_defaults: bool = True,
        metadata: Optional[dict] = None,
    ) -> None:
        if verbosity is not None:
            utils.setup_logging(verbosity)
        self.include_path = include_path
        self.metadata = metadata
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
        include_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        include_defaults: bool = True,
    ) -> dict[str, dict]:
        """Creates a dictionary of tasks indexes.

        :param include_path: Union[str, List] = None
            An additional path to be searched for tasks recursively.
            Can provide more than one such path as a list.
        :param include_defaults: bool = True
            If set to false, default tasks (those in lm_eval/tasks/) are not indexed.
        return
            Dictionary of task names as key and task metadata
        """
        if include_defaults:
            all_paths = [Path(__file__).parent]
        else:
            all_paths = []
        if include_path is not None:
            if isinstance(include_path, (str, Path)):
                include_path = [include_path]
            # Convert all paths to Path objects
            all_paths.extend(Path(p) for p in include_path)

        task_index = {}
        for task_dir in all_paths:
            tasks = self._get_task_and_group(task_dir)
            task_index = {**tasks, **task_index}

        return task_index

    @property
    def all_tasks(self) -> List[str]:
        return self._all_tasks

    @property
    def all_groups(self) -> List[str]:
        return self._all_groups

    @property
    def all_subtasks(self) -> List[str]:
        return self._all_subtasks

    @property
    def all_tags(self) -> List[str]:
        return self._all_tags

    @property
    def task_index(self) -> Dict[str, Dict[str, Union[str, int, List[str]]]]:
        return self._task_index

    def list_all_tasks(
        self,
        list_groups: bool = True,
        list_tags: bool = True,
        list_subtasks: bool = True,
    ) -> str:
        """
        Return a Markdown table (as a string) listing groups, tags and/or subtasks
        known to this TaskManager.  Safe for configs whose yaml_path is -1 and for
        task configs whose `include:` is a list.
        """
        from pytablewriter import MarkdownTableWriter

        # ------------------------------------------------------------------ helpers
        def sanitize_path(path: str) -> str:
            # print a relative path for anything inside lm_eval/tasks/
            # path_str = str(path)
            if "lm_eval/tasks/" in path:
                return "lm_eval/tasks/" + path.split("lm_eval/tasks/")[-1]
            return path

        def first_output_type_from_includes(cfg: dict, base: Path) -> str:
            """Walk cfg['include'] (string or list) and return the first
            include that itself specifies an output_type."""
            inc_raw = cfg.get("include")
            if not inc_raw:
                return ""

            inc_list = inc_raw if isinstance(inc_raw, list) else [inc_raw]
            for inc in inc_list:
                inc_path = Path(inc)
                if not inc_path.is_absolute():  # treat as relative include
                    inc_path = base.parent / inc_path
                try:
                    inc_cfg = load_yaml_config(inc_path, mode="simple")
                except FileNotFoundError:
                    continue
                if "output_type" in inc_cfg:
                    return inc_cfg["output_type"]
            return ""

        # -------------------------------------------------------------- GROUP table
        group_table = MarkdownTableWriter()
        group_table.headers = ["Group", "Config Location"]
        group_table.value_matrix = [
            [
                g,
                "---"
                if self.task_index[g]["yaml_path"] == -1
                else sanitize_path(self.task_index[g]["yaml_path"]),
            ]
            for g in self.all_groups
        ]

        # ---------------------------------------------------------------- TAG table
        tag_table = MarkdownTableWriter()
        tag_table.headers = ["Tag"]
        tag_table.value_matrix = [[t] for t in self.all_tags]

        # ------------------------------------------------------------ SUBTASK table
        subtask_table = MarkdownTableWriter()
        subtask_table.headers = ["Task", "Config Location", "Output Type"]
        st_values: list[list[str]] = []

        for t in self.all_subtasks:
            raw_path = self.task_index[t]["yaml_path"]

            if raw_path == -1:
                # python-only task or generated at runtime
                display_path = "---"
                output_type = ""
            else:
                path_obj = Path(raw_path)
                display_path = sanitize_path(str(path_obj))

                # load minimal YAML to discover output_type
                cfg = load_yaml_config(path_obj, mode="simple")
                if "output_type" in cfg:
                    output_type = cfg["output_type"]
                else:
                    output_type = first_output_type_from_includes(cfg, path_obj)

            st_values.append([t, display_path, output_type])

        subtask_table.value_matrix = st_values

        # ------------------------------------------------------------- final string
        parts: list[str] = ["\n"]
        if list_groups:
            parts.append(group_table.dumps())
            parts.append("\n")
        if list_tags:
            parts.append(tag_table.dumps())
            parts.append("\n")
        if list_subtasks:
            parts.append(subtask_table.dumps())
            parts.append("\n")

        return "".join(parts)

    def match_tasks(self, task_list: list[str]) -> list[str]:
        return utils.pattern_match(task_list, self.all_tasks)

    def _name_is_registered(self, name: str) -> bool:
        return name in self.all_tasks

    def _name_is_task(self, name: str) -> bool:
        return (
            self._name_is_registered(name) and self.task_index[name]["type"] == "task"
        )

    def _name_is_tag(self, name: str) -> bool:
        return self._name_is_registered(name) and self.task_index[name]["type"] == "tag"

    def _name_is_group(self, name: str) -> bool:
        return (
            self._name_is_registered(name) and self.task_index[name]["type"] == "group"
        )

    def _name_is_python_task(self, name: str) -> bool:
        return (
            self._name_is_registered(name)
            and self.task_index[name]["type"] == "python_task"
        )

    def _config_is_task(self, config: dict) -> bool:
        return "task" in config and isinstance(config["task"], str)

    def _config_is_group(self, config: dict) -> bool:
        return "task" in config and isinstance(config["task"], list)

    def _config_is_python_task(self, config: dict) -> bool:
        return "class" in config

    def _config_is_task_list(self, config: dict) -> bool:
        return "task_list" in config and isinstance(config["task_list"], list)

    def _get_yaml_path(self, name: str) -> Union[str, int]:
        if name not in self.task_index:
            raise ValueError
        return self.task_index[name]["yaml_path"]

    def _get_config(self, name: str) -> Dict:
        if name not in self.task_index:
            raise ValueError
        yaml_path = self._get_yaml_path(name)
        if yaml_path == -1:
            return {}
        else:
            return load_yaml_config(Path(yaml_path), mode="full")

    def _get_tasklist(self, name: str) -> Union[List[str], int]:
        if self._name_is_task(name):
            raise ValueError
        return self.task_index[name]["task"]

    def _register_task(
        self,
        task_name: str,
        task_type: str,
        yaml_path: str,
        tasks_and_groups: Dict[str, Dict],
        config: Optional[Dict] = None,
        populate_tags_fn: Optional[callable] = None,
    ) -> None:
        """Helper method to register a task in the tasks_and_groups dict"""
        tasks_and_groups[task_name] = {
            "type": task_type,
            "yaml_path": yaml_path,
        }
        # Only populate tags for configs that support it (not groups)
        if config and task_type != "group" and populate_tags_fn:
            populate_tags_fn(config, task_name, tasks_and_groups)

    def _merge_task_configs(
        self, base_config: Dict, task_specific_config: Dict, task_name: str
    ) -> Dict:
        """Merge base config with task-specific overrides for task_list configs"""
        if task_specific_config:
            task_specific_config = task_specific_config.copy()
            task_specific_config.pop("task", None)
            return {**base_config, **task_specific_config, "task": task_name}
        return {**base_config, "task": task_name}

    def _process_tag_subtasks(
        self, tag_name: str, update_config: Optional[Dict] = None
    ) -> Dict:
        """Process subtasks for a tag and return loaded tasks"""
        subtask_list = self._get_tasklist(tag_name)
        fn = partial(
            self._load_individual_task_or_group,
            update_config=update_config,
        )
        return dict(collections.ChainMap(*map(fn, reversed(subtask_list))))

    def _process_alias(self, config: Dict, group: Optional[str] = None) -> Dict:
        # If the group is not the same as the original
        # group which the group alias was intended for,
        # Set the group_alias to None instead.
        if ("group_alias" in config) and ("group" in config) and group is not None:
            if config["group"] != group:
                config["group_alias"] = None
        return config

    def _class_has_config_in_constructor(self, cls) -> bool:
        constructor = getattr(cls, "__init__", None)
        return (
            "config" in inspect.signature(constructor).parameters
            if constructor
            else False
        )

    def _load_individual_task_or_group(
        self,
        name_or_config: Optional[Union[str, Dict]] = None,
        parent_name: Optional[str] = None,
        update_config: Optional[Dict] = None,
    ) -> Mapping:
        def _load_task(
            config: Dict, task: str, yaml_path: Optional[str] = None
        ) -> Dict[str, Union[ConfigurableTask, Task]]:
            if "include" in config:
                # Store the task name to preserve it after include processing
                original_task_name = config.get("task", task)

                config = {
                    **load_yaml_config(
                        yaml_path=Path(yaml_path),
                        yaml_config={"include": config.pop("include")},
                        mode="full" if yaml_path else "simple",
                    ),
                    **config,
                    "task": original_task_name,
                }

                # Ensure the task name from the group config is preserved
                # This prevents tasks with the same include from being treated as duplicates

            if self._config_is_python_task(config):
                if self._class_has_config_in_constructor(config["class"]):
                    task_object = config["class"](config=config)
                else:
                    task_object = config["class"]()
                if isinstance(task_object, ConfigurableTask):
                    # very scuffed: set task name here. TODO: fixme?
                    task_object.config.task = task
            else:
                if self.metadata is not None:
                    config["metadata"] = config.get("metadata", {}) | self.metadata
                else:
                    config["metadata"] = config.get("metadata", {})
                task_object = ConfigurableTask(config=config)

            return {task: task_object}

        def _get_group_and_subtask_from_config(
            config: Dict,
        ) -> tuple[ConfigurableGroup, List[str]]:
            if self.metadata is not None:
                config["metadata"] = config.get("metadata", {}) | self.metadata
            group_name = ConfigurableGroup(config=config)
            subtask_list = []
            for task in group_name.config["task"]:
                if isinstance(task, str) and self._name_is_tag(task):
                    subtask_list.extend(self._get_tasklist(task))
                else:
                    subtask_list.append(task)
            return group_name, subtask_list

        def _process_group_config(
            config: Dict, update_config: Optional[Dict] = None
        ) -> tuple[Dict, Optional[Dict]]:
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
                # Get the yaml_path for this task
                yaml_path = self._get_yaml_path(name_or_config)
                task_config = self._get_config(name_or_config)

                # Handle task_list configs
                if "task_list" in task_config:
                    # Find the specific task entry
                    task_specific_config = None
                    for task_entry in task_config["task_list"]:
                        if (
                            isinstance(task_entry, dict)
                            and task_entry.get("task") == name_or_config
                        ):
                            task_specific_config = task_entry
                            break

                    if task_specific_config:
                        # Create base config without task_list
                        base_config = {
                            k: v for k, v in task_config.items() if k != "task_list"
                        }
                        # Merge using helper method
                        task_config = self._merge_task_configs(
                            base_config, task_specific_config, name_or_config
                        )
                    else:
                        # Task not found in task_list, shouldn't happen if indexing worked correctly
                        eval_logger.warning(
                            f"Task {name_or_config} not found in task_list"
                        )
                        task_config = {"task": name_or_config}

                return _load_task(task_config, task=name_or_config, yaml_path=yaml_path)
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
                        return self._process_tag_subtasks(
                            name_or_config,
                            name_or_config
                            if isinstance(name_or_config, dict)
                            else None,
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
                    return self._process_tag_subtasks(name, name_or_config)
                else:
                    yaml_path = None
                    if self._name_is_registered(name):
                        yaml_path = self._get_yaml_path(name)
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
                    return _load_task(task_config, task=name, yaml_path=yaml_path)
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

    def load_task_or_group(
        self, task_list: Optional[Union[str, List[str]]] = None
    ) -> Dict:
        """Loads a dictionary of task objects from a list

        :param task_list: Union[str, list] = None
            Single string or list of string of task names to be loaded

        :return
            Dictionary of task objects
        """
        if isinstance(task_list, str):
            task_list = [task_list]

        all_loaded_tasks = dict(
            collections.ChainMap(
                *map(
                    lambda task: self._load_individual_task_or_group(task),
                    task_list,
                )
            )
        )
        return all_loaded_tasks

    def load_config(self, config: Dict) -> Mapping:
        return self._load_individual_task_or_group(config)

    def _get_task_and_group(self, task_dir: Union[str, Path]) -> Dict[str, Dict]:
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

        def _populate_tags_and_groups(
            config: Dict, task: str, tasks_and_groups: Dict[str, Dict]
        ) -> None:
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
                        eval_logger.info(
                            f"The tag '{tag}' is already registered as a group, this tag will not be registered. "
                            "This may affect tasks you want to call."
                        )
                        break
                    else:
                        tasks_and_groups[tag]["task"].append(task)

        # TODO: remove group in next release
        # ignore_dirs = [
        #     "__pycache__",
        #     ".ipynb_checkpoints",
        # ]
        tasks_and_groups = collections.defaultdict()
        task_dir_path = Path(task_dir)

        for yaml_path in iter_yaml_files(task_dir_path):
            try:
                config = load_yaml_config(
                    yaml_path, mode="simple", resolve_includes=False
                )
            except (FileNotFoundError, YAMLError, OSError) as err:
                eval_logger.debug(f"File {yaml_path} could not be loaded ({err})")
                continue
            if self._config_is_python_task(config):
                # This is a python class config
                task = config["task"]
                self._register_task(
                    task,
                    "python_task",
                    str(yaml_path),
                    tasks_and_groups,
                    config,
                    _populate_tags_and_groups,
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
                    "yaml_path": str(yaml_path),
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
                self._register_task(
                    task,
                    "task",
                    str(yaml_path),
                    tasks_and_groups,
                    config,
                    _populate_tags_and_groups,
                )
            elif self._config_is_task_list(config):
                # This is a task_list config
                for task_entry in config["task_list"]:
                    if isinstance(task_entry, dict) and "task" in task_entry:
                        task_name = task_entry["task"]
                        self._register_task(
                            task_name,
                            "task",
                            str(yaml_path),
                            tasks_and_groups,
                            config,
                            _populate_tags_and_groups,
                        )
            else:
                eval_logger.debug(f"File {yaml_path} could not be loaded")

        return tasks_and_groups


def get_task_name_from_config(task_config: Dict[str, str]) -> str:
    if "task" in task_config:
        return task_config["task"]
    if "dataset_name" in task_config:
        return "{dataset_path}_{dataset_name}".format(**task_config)
    else:
        return "{dataset_path}".format(**task_config)


def get_task_name_from_object(task_object: Union[ConfigurableTask, Task]) -> str:
    if hasattr(task_object, "config"):
        return task_object._config["task"]

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def _check_duplicates(task_dict: Dict[str, List[str]]) -> None:
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
) -> Dict[str, Union[ConfigurableTask, Task]]:
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
