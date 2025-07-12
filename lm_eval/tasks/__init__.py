"""
Task Management Module for LM Evaluation Harness.

This module provides comprehensive task discovery, loading, and management functionality
for the LM Evaluation Harness. It handles YAML configuration parsing with include support,
dynamic function importing, and task indexing across multiple directories.

Key Components:
- TaskManager: Main class for task discovery and management
- YAML configuration loading with !function tag support
- Task, group, and tag indexing
- Include resolution with cycle detection
- Caching for performance optimization

Example:
    Basic usage::

        task_manager = TaskManager()
        all_tasks = task_manager.all_tasks
        task_config = task_manager._get_config("hellaswag")

    Custom task paths::

        task_manager = TaskManager(
            include_path="/path/to/custom/tasks",
            include_defaults=True
        )
"""

import collections
import functools
import importlib.util
import inspect
import logging
import sys
from functools import partial
from glob import iglob
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Mapping,
    Optional,
    Union,
)

import yaml
from yaml import YAMLError

from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval.evaluator_utils import get_subtask_list
from lm_eval.utils import pattern_match, setup_logging


if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask, Task

eval_logger = logging.getLogger(__name__)

#: List of configuration keys that are specific to groups only
GROUP_ONLY_KEYS = list(GroupConfig().to_dict().keys())

#: Base YAML loader class - uses C loader if available for performance
_Base = yaml.CLoader if getattr(yaml, "__with_libyaml__", False) else yaml.FullLoader

#: Directory names to ignore during task discovery
_IGNORE_DIRS = (
    "__pycache__",
    ".ipynb_checkpoints",
)


def ignore_constructor(loader: yaml.Loader, node: yaml.Node) -> None:
    """YAML constructor that ignores !function tags during simple parsing."""
    return None


@functools.lru_cache(maxsize=2048)  # ← reuse per (directory, simple) pair
def _make_loader(yaml_dir: Path, simple: bool = False) -> type[yaml.Loader]:
    """
    Return a custom YAML Loader class bound to *yaml_dir*.

    yaml_dir
        Directory that holds the YAML file being parsed.
        We capture it so that !function look-ups can resolve relative
        Python files like  my_utils.some_fn  ➜  yaml_dir / "my_utils.py".
    simple
        If True we ignore !function completely (used by `mode="simple"`),
        used on TaskManager init to index.
    """

    class Loader(_Base):
        """Dynamically-generated loader that knows its base directory."""

    # Register (or stub) the !function constructor **for this Loader only**
    if simple:
        yaml.add_constructor("!function", ignore_constructor, Loader=Loader)
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


@functools.lru_cache(maxsize=None)  # ← cache module objects
def _import_function(qualname: str, *, base_path: Path) -> Callable:
    """
    Dynamically import a function from a Python module relative to base_path.

    This function enables YAML files to reference Python functions using
    the !function tag. Supports dot notation for nested modules.

    Args:
        qualname: Qualified function name like "my_module.my_function"
        base_path: Base directory for resolving relative module paths

    Returns:
        The imported callable function

    Raises:
        ValueError: If qualname doesn't contain a module part

    Example:
        >>> func = _import_function("utils.custom_metric", base_path=Path("/tasks"))
        >>> result = func(predictions, references)
    """
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


@functools.lru_cache(maxsize=4096)
def _parse_yaml_file(path: Path, mode: str) -> dict:
    """
    Parse a single YAML file with the appropriate loader.

    Args:
        path: Path to the YAML file
        mode: Parsing mode ("full" or "simple")

    Returns:
        Parsed YAML configuration as dictionary
    """
    loader_cls = _make_loader(path.parent, simple=(mode == "simple"))
    with path.open("rb") as fh:
        return yaml.load(fh, Loader=loader_cls)


@functools.lru_cache(maxsize=4096)
def _get_cached_config(yaml_path: Path, mode: str) -> dict:
    """Load and cache resolved YAML configs"""
    # Parse the YAML file
    yaml_config = _parse_yaml_file(yaml_path, mode)
    yaml_dir = yaml_path.parent

    # Handle includes
    include = yaml_config.pop("include", None)
    if not include:
        return yaml_config

    include_paths = include if isinstance(include, list) else [include]
    final_cfg: dict = {}

    for inc in reversed(include_paths):
        if inc is None:
            continue
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = (yaml_dir / inc_path).resolve()
        # Recursive call will use the cache
        included = _get_cached_config(inc_path, mode)
        final_cfg.update(included)

    final_cfg.update(yaml_config)  # local keys win
    return final_cfg


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

        # ---------- fast-path: use LRU cached function ----------
        if yaml_config is None and resolve_includes:
            return _get_cached_config(yaml_path, mode)

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
    return final_cfg


def iter_yaml_files(root: Path, ignore=_IGNORE_DIRS) -> Generator[Path, Any, None]:
    """
    Recursively iterate over all YAML files in a directory tree.

    Excludes files in ignored directories like __pycache__ and .ipynb_checkpoints.

    Args:
        root: Root directory to search for YAML files

    Yields:
        Path objects for each discovered YAML file

    Example:
        >>> for yaml_file in iter_yaml_files(Path("tasks")):
        ...     print(f"Found task config: {yaml_file}")
    """
    for p in iglob("**/*.yaml", root_dir=root, recursive=True):
        # ignore check
        if Path(p).parts[0] in ignore:
            continue
        yield root / p


class TaskManager:
    """
    Central manager for task discovery, indexing, and loading.

    TaskManager scans directories for YAML task configurations and maintains
    an index of all available tasks, groups, and tags. It provides methods
    for listing, filtering, and loading tasks with their configurations.

    The manager supports:
    - Automatic discovery from default lm_eval/tasks/ directory
    - Custom task directories via include_path
    - Task grouping and tagging
    - Configuration inheritance via YAML includes
    - Caching for performance

    Attributes:
        include_path: Additional directories to search for tasks
        metadata: Global metadata to inject into all task configs
        task_group_map: Mapping of tasks to their parent groups

    Example:
        Basic usage::

            tm = TaskManager()
            print(f"Found {len(tm.all_tasks)} tasks")
            hellaswag_config = tm._get_config("hellaswag")

        With custom tasks::

            tm = TaskManager(
                include_path="/my/custom/tasks",
                verbosity="INFO"
            )
            custom_tasks = [t for t in tm.all_tasks if "custom" in t]
    """

    def __init__(
        self,
        verbosity: Optional[str] = None,
        include_path: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
        include_defaults: bool = True,
        metadata: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize the TaskManager.

        Args:
            verbosity: Logging verbosity level (DEBUG, INFO, WARNING, ERROR)
            include_path: Additional path(s) to search for tasks. Can be a single
                         path or list of paths.
            include_defaults: Whether to include default tasks from lm_eval/tasks/
            metadata: Global metadata dictionary to inject into all task configs
        """
        if verbosity is not None:
            setup_logging(verbosity)
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
        include_path: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
        include_defaults: bool = True,
    ) -> dict[str, dict]:
        """Creates a dictionary of tasks indexes.

        :param include_path: Union[str, list] = None
            An additional path to be searched for tasks recursively.
            Can provide more than one such path as a list.
        :param include_defaults: bool = True
            If set to false, default tasks (those in lm_eval/tasks/) are not indexed.
        return
            dictionary of task names as key and task metadata
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
    def all_tasks(self) -> list[str]:
        """Get sorted list of all task names (tasks, groups, and tags)."""
        return self._all_tasks

    @property
    def all_groups(self) -> list[str]:
        """Get sorted list of all group names."""
        return self._all_groups

    @property
    def all_subtasks(self) -> list[str]:
        """Get sorted list of all individual task names (excludes groups and tags)."""
        return self._all_subtasks

    @property
    def all_tags(self) -> list[str]:
        """Get sorted list of all tag names."""
        return self._all_tags

    @property
    def task_index(self) -> dict[str, dict[str, Union[str, int, list[str]]]]:
        """Get the complete task index with metadata for all tasks."""
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
                if inc:
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
        """Match task names using glob-style pattern matching."""
        return pattern_match(task_list, self.all_tasks)

    def _name_is_registered(self, name: str) -> bool:
        """Check if a name is registered in the task index."""
        return name in self.all_tasks

    def _name_is_task(self, name: str) -> bool:
        """Check if a name refers to an individual task (not group or tag)."""
        return (
            self._name_is_registered(name) and self.task_index[name]["type"] == "task"
        )

    def _name_is_tag(self, name: str) -> bool:
        """Check if a name refers to a tag."""
        return self._name_is_registered(name) and self.task_index[name]["type"] == "tag"

    def _name_is_group(self, name: str) -> bool:
        """Check if a name refers to a group."""
        return (
            self._name_is_registered(name) and self.task_index[name]["type"] == "group"
        )

    def _name_is_python_task(self, name: str) -> bool:
        """Check if a name refers to a Python-defined task."""
        return (
            self._name_is_registered(name)
            and self.task_index[name]["type"] == "python_task"
        )

    def _config_is_task(self, config: dict) -> bool:
        """Check if a config dictionary defines a single task."""
        return "task" in config and isinstance(config["task"], str)

    def _config_is_group(self, config: dict) -> bool:
        """Check if a config dictionary defines a group of tasks."""
        return "task" in config and isinstance(config["task"], list)

    def _config_is_python_task(self, config: dict) -> bool:
        """Check if a config dictionary defines a Python class-based task."""
        return "class" in config

    def _config_is_task_list(self, config: dict) -> bool:
        """Check if a config dictionary defines a task list."""
        return "task_list" in config and isinstance(config["task_list"], list)

    def _get_yaml_path(self, name: str) -> Union[str, int]:
        """
        Get the YAML file path for a registered task.

        Args:
            name: Task name

        Returns:
            Path to YAML file, or -1 for Python-only tasks

        Raises:
            ValueError: If task name is not registered
        """
        if name not in self.task_index:
            raise ValueError
        return self.task_index[name]["yaml_path"]

    def _get_config(self, name: str) -> dict:
        """
        Load the full configuration for a registered task.

        Args:
            name: Task name

        Returns:
            Complete task configuration dictionary

        Raises:
            ValueError: If task name is not registered
        """
        if name not in self.task_index:
            raise ValueError
        yaml_path = self._get_yaml_path(name)
        if yaml_path == -1:
            return {}
        else:
            return load_yaml_config(Path(yaml_path), mode="full")

    def _get_tasklist(self, name: str) -> Union[list[str], int]:
        """
        Get the task list for a group or tag.

        Args:
            name: Group or tag name

        Returns:
            List of task names in the group/tag

        Raises:
            ValueError: If name refers to an individual task
        """
        if self._name_is_task(name):
            raise ValueError
        return self.task_index[name]["task"]

    def _register_task(
        self,
        task_name: str,
        task_type: str,
        yaml_path: str,
        tasks_and_groups: dict[str, dict],
        config: Optional[dict] = None,
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
        self, base_config: dict, task_specific_config: dict, task_name: str
    ) -> dict:
        """Merge base config with task-specific overrides for task_list configs"""
        if task_specific_config:
            task_specific_config = task_specific_config.copy()
            task_specific_config.pop("task", None)
            return {**base_config, **task_specific_config, "task": task_name}
        return {**base_config, "task": task_name}

    def _process_tag_subtasks(
        self, tag_name: str, update_config: Optional[dict] = None
    ) -> dict:
        """Process subtasks for a tag and return loaded tasks"""
        subtask_list = self._get_tasklist(tag_name)
        fn = partial(
            self._load_individual_task_or_group,
            update_config=update_config,
        )
        return dict(collections.ChainMap(*map(fn, reversed(subtask_list))))

    def _process_alias(self, config: dict, group: Optional[str] = None) -> dict:
        """
        Process group alias configuration.

        If the group is not the same as the original group which the group alias
        was intended for, set the group_alias to None instead.

        Args:
            config: Task configuration dictionary
            group: Group name to validate against

        Returns:
            Modified configuration with processed aliases
        """
        if ("group_alias" in config) and ("group" in config) and group is not None:
            if config["group"] != group:
                config["group_alias"] = None
        return config

    def _class_has_config_in_constructor(self, cls) -> bool:
        """
        Check if a class constructor accepts a 'config' parameter.

        Args:
            cls: Class to inspect

        Returns:
            True if constructor has 'config' parameter, False otherwise
        """
        constructor = getattr(cls, "__init__", None)
        return (
            "config" in inspect.signature(constructor).parameters
            if constructor
            else False
        )

    ###############################################################################
    # NEW: Refactored _load_individual_task_or_group and helper methods          #
    ###############################################################################

    def _create_task_object(
        self,
        cfg: dict,
        task_name: str,
        yaml_path: str | None,
    ) -> dict:
        """
        Instantiate a single task (ConfigurableTask **or** python-task) from *cfg*.
        Returns {task_name: task_object}.
        """
        from lm_eval.api.task import ConfigurableTask, Task  # local import avoids cycle

        # ---- include handling ---------------------------------------------------
        if "include" in cfg:
            # keep original name so include merging cannot clobber it
            orig_name = cfg.get("task", task_name)
            cfg = {
                **load_yaml_config(  # recurse once, cached
                    yaml_path=Path(yaml_path) if yaml_path else None,
                    yaml_config={"include": cfg.pop("include")},
                    mode="full" if yaml_path else "simple",
                ),
                **cfg,
                "task": orig_name,
            }

        # ---- metadata merge -----------------------------------------------------
        if self.metadata is not None:
            cfg["metadata"] = cfg.get("metadata", {}) | self.metadata
        else:
            cfg["metadata"] = cfg.get("metadata", {})

        # ---- python-task vs YAML-task -------------------------------------------
        if self._config_is_python_task(cfg):
            cls = cfg["class"]
            task_obj: Task
            if self._class_has_config_in_constructor(cls):
                task_obj = cls(config=cfg)
            else:
                task_obj = cls()
            # make sure name propagates when the class inherits ConfigurableTask
            if isinstance(task_obj, ConfigurableTask):  # type: ignore
                task_obj.config.task = task_name
        else:
            task_obj = ConfigurableTask(config=cfg)  # type: ignore

        return {task_name: task_obj}

    def _create_group_object(
        self,
        cfg: dict,
        parent_name: str | None = None,
    ) -> tuple[ConfigurableGroup, list[Union[str, dict]]]:
        """
        Build ConfigurableGroup and return (group_obj, subtask_names).
        Resolves tag expansion.
        """
        if self.metadata is not None:
            cfg["metadata"] = cfg.get("metadata", {}) | self.metadata

        grp = ConfigurableGroup(config=cfg)
        subtasks: list[Union[str, dict]] = []
        for t in grp.config["task"]:
            if isinstance(t, str) and self._name_is_tag(t):
                subtasks.extend(self._get_tasklist(t))
            else:
                subtasks.append(t)
        return grp, subtasks

    def _load_subtasks(
        self,
        subtasks: list[Union[str, dict]],
        parent_name: Union[str, ConfigurableGroup, None],
        update_config: dict | None,
    ) -> Mapping:
        """Return merged mapping of all subtasks, handling duplicates."""
        fn = functools.partial(
            self._load_individual_task_or_group,
            parent_name=parent_name,
            update_config=update_config,
        )
        return dict(collections.ChainMap(*map(fn, reversed(subtasks))))

    def _load_individual_task_or_group(
        self,
        payload: str | dict,
        *,
        parent_name: str | None = None,
        update_config: dict | None = None,
    ) -> Mapping:
        """
        Public helper that turns *payload* (str task/group/tag **or** dict config)
        into a nested Mapping of {name_or_group_obj: task_obj | sub_mapping}.
        """

        # ------------------------------------------------------------------ STRING
        if isinstance(payload, str):
            # If caller supplied extra overrides, treat as dict immediately
            if update_config:
                return self._load_individual_task_or_group(
                    {"task": payload, **update_config},
                    parent_name=parent_name,
                )

            # ------------ registered TASK (YAML or python) -----------------
            if self._name_is_task(payload) or self._name_is_python_task(payload):
                yaml_path = self._get_yaml_path(payload)
                cfg = self._get_config(payload)

                # task_list configs: extract the per-task override ------------
                if "task_list" in cfg:
                    override = next(
                        (
                            entry
                            for entry in cfg["task_list"]
                            if isinstance(entry, dict) and entry.get("task") == payload
                        ),
                        None,
                    )
                    base = {k: v for k, v in cfg.items() if k != "task_list"}
                    if override:
                        cfg = {**base, **override, "task": payload}
                return self._create_task_object(cfg, payload, yaml_path)

            # ------------ registered GROUP ----------------------------------
            if self._name_is_group(payload):
                group_cfg = self._get_config(payload)
                grp_only = {k: v for k, v in group_cfg.items() if k in GROUP_ONLY_KEYS}
                grp_obj, subtasks = self._create_group_object(grp_only, parent_name)
                return {
                    grp_obj: self._load_subtasks(subtasks, grp_obj, update_config=None)
                }

            # ------------ registered TAG ------------------------------------
            if self._name_is_tag(payload):
                return self._process_tag_subtasks(payload, update_config=None)

            raise ValueError(f"Unknown task / group / tag name: {payload!r}")

        # ------------------------------------------------------------------- DICT
        if isinstance(payload, dict):
            # ------------------ simple 'task: name' dict --------------------
            if self._config_is_task(payload):
                name = payload["task"]
                # override existing registered YAML if exists
                if self._name_is_registered(name):
                    base_cfg = self._get_config(name)
                    yaml_path = self._get_yaml_path(name)
                    merged = {**base_cfg, **payload}
                else:
                    merged = payload
                    yaml_path = None

                # duplicate-naming guard when inside a group
                if parent_name is not None:
                    count = len(
                        [
                            n
                            for n in self.task_group_map[parent_name]
                            if n.startswith(name)
                        ]
                    )
                    if count:
                        name = f"{name}-{count}"
                    self.task_group_map[parent_name].append(name)

                return self._create_task_object(merged, name, yaml_path)

            # ----------------- literal group dict (task: [...]) -------------
            if self._config_is_group(payload):
                grp_cfg = {k: v for k, v in payload.items() if k in GROUP_ONLY_KEYS}
                sub_override = {
                    k: v for k, v in payload.items() if k not in GROUP_ONLY_KEYS
                } or None
                grp_obj, subtasks = self._create_group_object(grp_cfg, parent_name)
                return {grp_obj: self._load_subtasks(subtasks, grp_obj, sub_override)}

            # ----------------- python-task dict ('class': …) ----------------
            if self._config_is_python_task(payload):
                name = payload["task"]
                return self._create_task_object(payload, name, yaml_path=None)

        raise TypeError(
            f"_load_individual_task_or_group expected str | dict, got {type(payload)}"
        )

    def load_task_or_group(
        self, task_list: Optional[Union[str, list[str]]] = None
    ) -> dict:
        """
        Load multiple tasks or groups from a list of names.

        This is the main entry point for loading tasks. It handles lists
        of task names and delegates to _load_individual_task_or_group for
        each item, then merges the results.

        Args:
            task_list: Single task name or list of task names to load.
                      Can include individual tasks, groups, and tags.

        Returns:
            Dictionary mapping task/group names to loaded task objects.
            Results from all requested items are merged into a single dict.

        Example:
            Load multiple tasks::

                tasks = tm.load_task_or_group(["hellaswag", "arc_easy"])
                # Returns: {"hellaswag": Task1, "arc_easy": Task2}

            Load a group::

                tasks = tm.load_task_or_group("arc_group")
                # Returns: {"arc_group": {"arc_easy": Task1, "arc_challenge": Task2}}
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

    def load_config(self, config: dict) -> Mapping:
        """
        Load a task from an inline configuration dictionary.

        Args:
            config: Configuration dictionary defining the task

        Returns:
            Mapping of task name to loaded task object

        Example:
            >>> config = {"task": "hellaswag", "num_fewshot": 5}
            >>> task_dict = tm.load_config(config)
        """
        return self._load_individual_task_or_group(config)

    def _get_task_and_group(self, task_dir: Union[str, Path]) -> dict[str, dict]:
        """
        Scan a directory for task configurations and build an index.

        Creates a dictionary of task metadata by recursively scanning for
        YAML files and parsing their configurations. This method handles:
        - Regular task configs with 'task' key
        - Python class-based tasks with 'class' key
        - Group configs with 'group' key
        - Task list configs with 'task_list' key
        - Tag extraction and registration

        Args:
            task_dir: Directory path to scan for YAML task configurations

        Returns:
            Dictionary mapping task names to metadata dictionaries.
            Each metadata dict contains:
            - 'type': One of 'task', 'python_task', 'group', 'tag'
            - 'yaml_path': Path to source YAML file (or -1 for generated entries)
            - 'task': For groups/tags, list of constituent task names

        Note:
            This method is called during TaskManager initialization to build
            the master task index. It uses 'simple' parsing mode for performance.
        """

        def _populate_tags_and_groups(
            config: dict, task: str, tasks_and_groups: dict[str, dict]
        ) -> None:
            """
            Extract and register tags from a task configuration.

            Tags allow grouping tasks by theme or category. This function
            processes the 'tag' field in task configs and maintains tag
            indices for quick lookup.

            Args:
                config: Task configuration dictionary
                task: Name of the task being processed
                tasks_and_groups: Master index to update with tag information
            """
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


def get_task_name_from_config(task_config: dict[str, str]) -> str:
    """
    Extract a task name from a configuration dictionary.

    Determines the canonical name for a task based on its configuration,
    with fallback strategies for different config formats.

    Args:
        task_config: Task configuration dictionary

    Returns:
        String name for the task

    Example:
        >>> config = {"task": "hellaswag", "num_fewshot": 5}
        >>> get_task_name_from_config(config)
        'hellaswag'

        >>> config = {"dataset_path": "custom", "dataset_name": "mytask"}
        >>> get_task_name_from_config(config)
        'custom_mytask'
    """
    if "task" in task_config:
        return task_config["task"]
    if "dataset_name" in task_config:
        return "{dataset_path}_{dataset_name}".format(**task_config)
    else:
        return "{dataset_path}".format(**task_config)


def get_task_name_from_object(task_object: Union["ConfigurableTask", "Task"]) -> str:
    """
    Extract the name from an instantiated task object.

    Handles both ConfigurableTask and legacy Task objects with different
    attribute conventions for storing the task name.

    Args:
        task_object: An instantiated task object

    Returns:
        String name of the task

    Example:
        >>> task = ConfigurableTask(config={"task": "hellaswag"})
        >>> get_task_name_from_object(task)
        'hellaswag'
    """
    if hasattr(task_object, "config"):
        return task_object._config["task"]

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def _check_duplicates(task_dict: dict[str, list[str]]) -> None:
    """
    Validate that no tasks appear in multiple groups simultaneously.

    Helper function used to prevent conflicts when multiple groups claim
    the same constituent task. This could lead to ambiguous configuration
    like conflicting num_fewshot values.

    Args:
        task_dict: Dictionary mapping group names to lists of subtask names

    Raises:
        ValueError: If any tasks appear in multiple groups

    Example:
        >>> task_dict = {
        ...     "group1": ["task_a", "task_b"],
        ...     "group2": ["task_b", "task_c"]  # task_b appears twice!
        ... }
        >>> _check_duplicates(task_dict)  # Raises ValueError
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
    task_name_list: Union[str, list[Union[str, dict, "Task"]]],
    task_manager: Optional[TaskManager] = None,
) -> dict[str, Union["ConfigurableTask", "Task"]]:
    """
    Create a dictionary of task objects from mixed input types.

    This is the main public API for loading tasks. It accepts various input
    formats (names, configs, objects) and returns a unified dictionary of
    instantiated task objects ready for evaluation.

    The function handles:
    - String task names (looked up via TaskManager)
    - Configuration dictionaries (processed as inline configs)
    - Pre-instantiated Task objects (used as-is)
    - Validation to prevent conflicting group memberships

    Args:
        task_name_list: Mixed list of task specifications:
                       - str: Task name to look up
                       - dict: Inline task configuration
                       - Task: Pre-instantiated task object
        task_manager: TaskManager instance for name resolution.
                     If None, creates a default TaskManager.

    Returns:
        Dictionary mapping task names to instantiated task objects.
        All tasks are ready for evaluation.

    Raises:
        TypeError: If task_name_list contains unsupported types
        ValueError: If there are conflicting group memberships

    Example:
        Mixed input types::

            tasks = get_task_dict([
                "hellaswag",                              # lookup by name
                {"task": "arc_easy", "num_fewshot": 5},   # inline config
                pre_existing_task_object                  # direct object
            ])

        Simple case::

            tasks = get_task_dict("hellaswag")
            # Returns: {"hellaswag": ConfigurableTask(...)}

        With custom TaskManager::

            tm = TaskManager(include_path="/custom/tasks")
            tasks = get_task_dict(["custom_task"], task_manager=tm)
    """
    from lm_eval.api.task import Task

    # Normalize input to list
    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    elif not isinstance(task_name_list, list):
        raise TypeError(
            f"Expected a 'str' or 'list' but received {type(task_name_list)}."
        )

    # Validate list items
    if not all(isinstance(task, (str, dict, Task)) for task in task_name_list):
        raise TypeError(
            "Expected all list items to be of types 'str', 'dict', or 'Task', but at least one entry did not match."
        )

    # Ensure we have a task manager
    if task_manager is None:
        task_manager = TaskManager()

    # Process all items
    final_task_dict = {}
    for task_spec in task_name_list:
        if isinstance(task_spec, Task):
            # Pre-instantiated task object
            task_name = get_task_name_from_object(task_spec)
            if task_name in final_task_dict:
                raise ValueError(f"Duplicate task name: {task_name}")
            final_task_dict[task_name] = task_spec
        else:
            # String or dict - use load_task_or_group
            result = task_manager.load_task_or_group(task_spec)
            # Check for duplicate names
            for name in result:
                if name in final_task_dict:
                    raise ValueError(f"Duplicate task name: {name}")
            final_task_dict.update(result)

    # Check for conflicting group memberships
    _check_duplicates(get_subtask_list(final_task_dict))

    return final_task_dict
