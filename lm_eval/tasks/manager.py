from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import NotRequired, TypedDict, deprecated

from lm_eval import utils
from lm_eval.api.group import Group
from lm_eval.api.task import Task
from lm_eval.tasks.factory import TaskFactory
from lm_eval.tasks.index import Entry, Kind, TaskIndex


if TYPE_CHECKING:
    from collections.abc import Iterable


class TaskDict(TypedDict):
    tasks: dict[str, Task]
    groups: NotRequired[dict[str, Group]]


class TaskManager:
    """Discovers, indexes, and loads evaluation tasks from YAML configs.

    Scans directories for task definitions and provides methods to load them
    by name, glob pattern, or inline config. Handles groups, tags, and task
    namespacing (e.g., "mmlu_humanities::formal_logic").
    """

    def __init__(
        self,
        verbosity: str | None = None,
        include_path: str | Path | list[str | Path] | None = None,
        include_defaults: bool = True,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Args:
            verbosity: Logging level (e.g., "INFO", "DEBUG")
            include_path: Custom paths to scan for task configs (takes precedence)
            include_defaults: Whether to include built-in tasks from lm_eval/tasks/
            metadata: Extra metadata to attach to all loaded tasks
        """
        if verbosity:
            warnings.warn(
                "The `verbosity` argument is deprecated. Use logging configuration instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.include_path = include_path
        self.metadata = metadata

        index = TaskIndex()
        self._factory = TaskFactory(meta=metadata)

        all_paths: list[Path] = []
        # Process include_path FIRST so user tasks take precedence over defaults
        if include_path:
            all_paths += [
                Path(p)
                for p in (
                    include_path
                    if isinstance(include_path, (list, tuple))
                    else [include_path]
                )
            ]
        if include_defaults:
            all_paths.append(Path(__file__).parent)

        self._index = index.build(all_paths)

        buckets = defaultdict(list)
        for k, e in self._index.items():
            buckets[e.kind].append(k)

        self._all_tasks = sorted(self._index.keys())
        self._all_subtasks = sorted(
            chain.from_iterable(buckets[k] for k in {Kind.TASK, Kind.PY_TASK})
        )
        self._all_groups = sorted(buckets[Kind.GROUP])
        self._all_tags = sorted(buckets[Kind.TAG])

    # ---------------------------------------------------------------- properties
    @property
    def all_tasks(self) -> list[str]:
        """All registered names (tasks, groups, tags)."""
        return self._all_tasks

    @property
    def all_groups(self) -> list[str]:
        """All group names (e.g., "mmlu", "arc")."""
        return self._all_groups

    @property
    def all_subtasks(self) -> list[str]:
        """All individual task names (YAML and Python tasks)."""
        return self._all_subtasks

    @property
    def all_tags(self) -> list[str]:
        """All tag names (e.g., "ai2_arc", "mmlu_humanities_tasks")."""
        return self._all_tags

    @property
    def task_index(self) -> dict[str, Entry]:
        """Raw index mapping names to Entry objects."""
        return self._index

    # ---------------------------------------------------------------- core API
    def _entry(self, name: str) -> Entry:
        """Get the Entry for a given task/group/tag name from the index."""
        if name not in self._index:
            raise KeyError(f"Unknown task/group/tag: {name}")
        return self._index[name]

    def load_spec(self, spec: str | dict[str, Any]) -> Task | Group | list[Task]:
        """Load a task/group/tag by name or with inline overrides.

        Args:
            spec: Task name (str) or dict with "task" key and overrides

        Returns:
            Dict mapping task names to task objects (nested for groups)
        """
        match spec:
            case str():
                entry = self._entry(spec)
                return self._factory.build(entry, overrides=None, registry=self._index)
            case dict():
                name = spec.get("task")
                if not name:
                    raise KeyError("Inline config dict must have a 'task' key")
                entry = self._entry(name)
                return self._factory.build(entry, overrides=spec, registry=self._index)
            case _:
                raise TypeError("spec must be str or dict")

    def load_config(self, config: str | dict[str, Any]) -> dict:
        """Load a task from an inline config dict."""
        spec = self.load_spec(config)
        return self._to_nested_dict(spec)

    def load(
        self,
        task_list: str
        | list[str | Task | Group | dict[str, Any]]
        | Task
        | Group
        | dict[str, Any],
    ) -> TaskDict:
        """Load tasks/groups and return organized result.

        Groups contain their children (Tasks and sub-Groups) directly.
        Tags expand to individual Tasks.

        Args:
            task_list: Single task name or list of task names

        Returns:
            Dict with:
            - tasks: {task_name: Task} flat dict of all leaf tasks
            - groups: {group_name: Group} flat dict of all groups
        """
        if not isinstance(task_list, list):
            task_list = [task_list]  # type: ignore

        # Build all requested items
        built: list[Task | Group] = []
        for spec in cast("Iterable", task_list):
            obj = self.load_spec(spec) if not isinstance(spec, (Task, Group)) else spec  # type:ignore[invalid-argument-type]
            # Tags return list[Task], flatten
            if isinstance(obj, list):
                obj = cast("list[Task]", obj)
                built.extend(obj)
            else:
                built.append(obj)

        # Flatten to task/group dicts
        tasks: dict[str, Task] = {}
        groups: dict[str, Group] = {}

        def collect(item: Task | Group) -> None:
            if isinstance(item, Group):
                groups[item.name] = item
                for task in item.get_all_tasks():
                    tasks[task.task_name] = task
                for subgroup in item.get_all_groups():
                    groups[subgroup.name] = subgroup
            else:
                tasks[item.task_name] = item

        for item in built:
            collect(item)

        return {"tasks": tasks, "groups": groups}

    def _to_nested_dict(self, obj: Task | Group | list) -> dict:
        """Convert Task | Group | list[Task] to legacy nested dict format.

        This adapter maintains backward compatibility with get_task_dict() and other
        consumers that expect the old nested dict format with ConfigurableGroup keys.
        """
        from lm_eval.api.group import ConfigurableGroup, GroupConfig

        # Handle list of tasks (from tag expansion)
        if isinstance(obj, list):
            result: dict[str, Any] = {}
            for task in obj:
                task = cast("Task", task)
                result[task.task_name] = task
            return result

        if isinstance(obj, Group):
            # Build nested children dict
            nested: dict[str, Any] = {}
            for child in obj:
                if isinstance(child, Group):
                    nested.update(self._to_nested_dict(child))
                else:
                    # Task
                    nested[child.task_name] = child

            # Wrap in ConfigurableGroup
            config = GroupConfig(group=obj.name, group_alias=obj.alias)
            cg = ConfigurableGroup(config=config)
            return {cg: nested}

        # Task - return flat dict
        return {obj.task_name: obj}

    @deprecated("load_task_or_group is deprecated, use load() instead")
    def load_task_or_group(self, task_list: str | list[str]) -> dict:
        """Load tasks/groups and return a merged dictionary.

        Args:
            task_list: Single task name or list of task names
        Returns:
            Dictionary of task objects (possibly nested for groups)
        """
        import collections

        if isinstance(task_list, str):
            task_list = [task_list]

        # Each load_spec call returns a dict (possibly nested for groups)
        # We merge them using ChainMap (like the original implementation)
        return dict(collections.ChainMap(*[self.load_config(s) for s in task_list]))

    # ---------------------------------------------------------------- name checks
    def _name_is_registered(self, name: str) -> bool:
        return name in self._index

    def _name_is_task(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.TASK

    def _name_is_tag(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.TAG

    def _name_is_group(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.GROUP

    def _name_is_python_task(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.PY_TASK

    # ---------------------------------------------------------------- utility
    def match_tasks(self, task_list: list[str]) -> list[str]:
        """Match task names using glob patterns."""
        return utils.pattern_match(task_list, self.all_tasks)

    def list_all_tasks(
        self,
        list_groups: bool = True,
        list_tags: bool = True,
        list_subtasks: bool = True,
    ) -> str:
        """Generate a markdown table listing all available tasks."""
        from pytablewriter import MarkdownTableWriter

        def sanitize_path(path):
            if path is None:
                return "---"
            path_str = str(path)
            if "lm_eval/tasks/" in path_str:
                return "lm_eval/tasks/" + path_str.split("lm_eval/tasks/")[-1]
            return path_str

        group_table = MarkdownTableWriter()
        group_table.headers = ["Group", "Config Location"]
        gt_values = []
        for g in self.all_groups:
            entry = self._index[g]
            path = sanitize_path(entry.yaml_path)
            gt_values.append([g, path])
        group_table.value_matrix = gt_values

        tag_table = MarkdownTableWriter()
        tag_table.headers = ["Tag"]
        tag_table.value_matrix = [[t] for t in self.all_tags]

        subtask_table = MarkdownTableWriter()
        subtask_table.headers = ["Task", "Config Location", "Output Type"]
        st_values = []
        for t in self.all_subtasks:
            entry = self._index[t]
            path = entry.yaml_path
            output_type = ""

            if path is not None:
                config = utils.load_yaml_config(str(path), mode="simple")
                if "output_type" in config:
                    output_type = config["output_type"]
                elif "include" in config:
                    include_path = str(path.parent / config["include"])
                    include_config = utils.load_yaml_config(include_path, mode="simple")
                    if "output_type" in include_config:
                        output_type = include_config["output_type"]

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


def get_task_dict(
    task_name_list: str | list[str | dict | Task],
    task_manager: TaskManager | None = None,
):
    """Helper to load multiple tasks into a dict. Creates TaskManager if not provided."""
    if not task_manager:
        task_manager = TaskManager()
    else:
        assert isinstance(task_manager, TaskManager)

    return {
        task_name: task_manager.load_config(task_name)
        if isinstance(task_name, str)
        else task_name
        for task_name in task_name_list
    }
