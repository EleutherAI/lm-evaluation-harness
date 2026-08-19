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
from lm_eval.tasks._factory import TaskFactory
from lm_eval.tasks._index import Entry, Kind, TaskIndex
from lm_eval.tasks._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class TaskDict(TypedDict):
    """Return type of :meth:`TaskManager.load`.

    Attributes:
        tasks: Flat mapping of task name to Task for every leaf task.
        groups: Flat mapping of group name to Group.
        group_map: Mapping of each group name to its direct child names (not recursive).
    """

    tasks: dict[str, Task]
    groups: NotRequired[dict[str, Group]]
    group_map: NotRequired[dict[str, list[str]]]


class TaskManager:
    """Central entry point for discovering and loading evaluation tasks.

    On construction, scans one or more directories for YAML task configs and
    builds an in-memory index of every known task, group, and tag.  Callers
    then use :meth:`load` to instantiate tasks by name, glob pattern, file
    path, or inline config dict.

    Example::

        tm = TaskManager(include_path="my_tasks/")
        result = tm.load(["mmlu", "hellaswag"])
        result["tasks"]   # {name: Task, ...}
        result["groups"]  # {name: Group, ...}
    """

    def __init__(
        self,
        verbosity: str | None = None,
        include_path: str | Path | list[str | Path] | None = None,
        include_defaults: bool = True,
        metadata: dict[str, dict[str, Any] | str] | None = None,
    ) -> None:
        """
        Args:
            verbosity: Logging level (e.g., "INFO", "DEBUG") (deprecated, use standard logging configuration instead)
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
        self._factory: TaskFactory = TaskFactory(meta=metadata)

        all_paths: list[Path] = []
        # Process defaults FIRST, then include_path (later paths can override earlier)
        if include_defaults:
            all_paths.append(Path(__file__).parent)
        if include_path:
            all_paths += [
                Path(p)
                for p in (
                    include_path
                    if isinstance(include_path, (list, tuple))
                    else [include_path]
                )
            ]

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
    def _entry(self, name: str) -> Entry | None:
        """Get the Entry for a given task/group/tag name from the index."""
        return self._index.get(name)

    def _load_spec(self, spec: str | dict[str, Any]) -> Task | Group | list[Task]:
        """Load a task/group/tag by name, file path, or inline config.

        Args:
            spec: Task name (str), YAML file path,
                  or dict with "task"/"group" key.

        Returns:
            Task, Group, or list[Task] (for tags)
        """
        match spec:
            # Registered name
            case str():
                # First check if it's a registered name (task/group/tag)
                entry = self._entry(spec)
                if entry:
                    return self._factory.build(
                        entry, overrides=None, registry=self._index
                    )
                # check if it's a path
                entry = TaskIndex.entry_from_path(Path(spec))
                if entry:
                    return self._factory.build(
                        entry, overrides=None, registry=self._index
                    )
                raise KeyError(
                    f"Spec '{spec}' is not a registered task/group/tag name or valid YAML path"
                )

            case dict():
                _entry = TaskIndex.entry_from_config(spec)
                if _entry:
                    return self._factory.build(
                        _entry, overrides=spec, registry=self._index
                    )
                raise ValueError(
                    "Invalid config dict: must contain 'task' or 'group' key"
                )
            case _:
                raise TypeError("spec must be str or dict")

    def load(
        self,
        task_list: str
        | Sequence[str]
        | Sequence[str | Task | Group | dict[str, Any]]
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
            - group_map: {group_name: [child_names]}
        """
        if not isinstance(task_list, list):
            task_list = [task_list]  # type: ignore

        # Build all requested items
        built: list[Task | Group] = []
        for spec in cast("Iterable", task_list):
            obj = self._load_spec(spec) if not isinstance(spec, (Task, Group)) else spec  # type:ignore[invalid-argument-type]
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
        self._check_duplicates(built)
        return {
            "tasks": tasks,
            "groups": groups,
            "group_map": {g.name: g.child_names for g in groups.values()}
            if groups
            else {},
        }

    @deprecated("Use TaskManager.load(), which returns flat dicts of tasks and groups.")
    def load_task_or_group(self, task_list: str | list[str | dict]) -> dict:
        """Legacy loader that returns the old nested-dict format.

        Wraps :meth:`load` but converts the result into ``{ConfigurableGroup: {task_name: Task, ...}, ...}``
        dicts expected by callers.  New code should use :meth:`load` instead.

        Args:
            task_list: Single task name or list of task names/dicts.

        Returns:
            Nested dict keyed by :class:`ConfigurableGroup` (for groups) or
            ``task_name`` (for standalone tasks), with leaf values being
            :class:`Task` instances.
        """
        import collections

        from lm_eval.api.group import ConfigurableGroup

        if isinstance(task_list, str):
            task_list = [task_list]

        def _to_nested(obj: Task | Group | list[Task]) -> dict:
            """Convert Task | Group | list[Task] to legacy nested dict format."""
            if isinstance(obj, list):
                return {t.task_name: t for t in obj}  # type:ignore
            if isinstance(obj, Group):
                nested: dict[str, Any] = {}
                for child in obj:
                    if isinstance(child, Group):
                        nested.update(_to_nested(child))
                    else:
                        nested[child.task_name] = child
                cg = ConfigurableGroup.from_group(obj)
                return {cg: nested}
            return {obj.task_name: obj}

        return dict(
            collections.ChainMap(*[_to_nested(self._load_spec(s)) for s in task_list])
        )

    @staticmethod
    def _check_duplicates(built: list[Task | Group]) -> None:
        """Check that no task appears in more than one top-level item.

        For each top-level item (Task or Group), collect all leaf task names.
        If any task name appears in multiple top-level items, raise an error.
        This catches cases like requesting ["group_a", "group_b"] where both
        groups contain the same task, or requesting a task both standalone
        and as part of a group.
        """
        seen: dict[str, str] = {}  # task_name -> source name
        for item in built:
            if isinstance(item, Group):
                source = item.name
                task_names = [t.task_name for t in item.get_all_tasks()]
            else:
                source = item.task_name
                task_names = [item.task_name]

            for name in task_names:
                if name in seen and seen[name] != source:
                    raise ValueError(
                        f"Duplicate task '{name}': found in both '{seen[name]}' and '{source}'"
                    )
                seen[name] = source

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
                config = load_yaml(path, resolve_func=False, recursive=True)
                if "output_type" in config:
                    output_type = config["output_type"]

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
