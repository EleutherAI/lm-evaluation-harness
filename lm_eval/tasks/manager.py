from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import NotRequired, TypedDict, deprecated

from lm_eval import utils
from lm_eval.api.group import Group

from ._factory import TaskFactory
from ._index import Kind, TaskIndex
from ._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lm_eval.api.task import Task

    from ._index import Entry


eval_logger = logging.getLogger(__name__)


class TaskDict(TypedDict):
    """Return type of [TaskManager.load][TaskManager.load].

    Example:
        ```python
        loaded = task_manager.load(["mmlu", "arc_easy"])
        loaded["tasks"]  # {"mmlu_abstract_algebra": Task, "arc_easy": Task, ...}
        loaded["groups"]  # {"mmlu": Group}
        loaded["group_map"]  # {"mmlu": ["mmlu_abstract_algebra", ...]}
        ```
    """

    tasks: dict[str, Task]
    """Flat mapping of every leaf task name to its [Task][lm_eval.api.task.Task]."""

    groups: NotRequired[dict[str, Group]]
    """Flat mapping of every group name to its [Group][lm_eval.api.group.Group]."""

    group_map: NotRequired[dict[str, list[str]]]
    """Each group's direct children (not recursive)."""


class TaskManager:
    """Central entry point for discovering and loading evaluation tasks.

    Scans directories for YAML task configs and builds an in-memory index
    of every known task, group, and tag. Use [load][.load] to instantiate
    tasks by name, glob, file path, or override dict.

    Args:
        verbosity: Deprecated — use standard logging instead.
        include_path: Extra directories to scan (take precedence over defaults).
        include_defaults: Scan built-in ``lm_eval/tasks/`` directory.
        metadata: Attached to every loaded task (e.g. model args).

    Example:
        ```python
        tm = TaskManager(include_path="my_tasks/")
        loaded = tm.load(["mmlu", "hellaswag"])
        loaded["tasks"]  # {"mmlu_..": Task, "hellaswag": Task, ...}
        loaded["groups"]  # {"mmlu": Group}
        ```
    """

    def __init__(
        self,
        verbosity: str | None = None,
        include_path: str | Path | list[str | Path] | None = None,
        include_defaults: bool = True,
        metadata: dict[str, dict[str, Any] | str] | None = None,
    ) -> None:
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
    def _task_index(self) -> dict[str, Entry]:
        """Raw index mapping names to Entry objects."""
        return self._index

    # ---------------------------------------------------------------- core API
    def _entry(self, name: str) -> Entry | None:
        """Get the Entry for a given task/group/tag name from the index."""
        return self._index.get(name)

    def _resolve_path(
        self, spec: str, overrides: dict[str, Any] | None = None
    ) -> Task | Group:
        """Resolve a ``::``-separated path to an inline subgroup or task.

        Loads the root group from the index, then walks down the hierarchy
        using each ``::``-delimited segment.  Children are keyed by their
        full namespaced name (e.g. ``"parent::child"``), which is exactly
        what ``"::".join(parts[:i+1])`` produces at each level.

        Args:
            spec: A path like ``"group::subgroup"`` or ``"group::sub::task"``.
            overrides: Runtime overrides propagated when building the root group.

        Returns:
            The Task or Group at the end of the path.
        """
        parts = spec.split("::")
        root_name = parts[0]

        root = self._load_spec(root_name, overrides=overrides)
        if not isinstance(root, Group):
            raise KeyError(f"Root '{root_name}' in path '{spec}' is not a group")

        current: Task | Group = root
        for i in range(1, len(parts)):
            if not isinstance(current, Group):
                path_so_far = "::".join(parts[:i])
                raise KeyError(
                    f"Cannot navigate into '{parts[i]}': "
                    f"'{path_so_far}' is a task, not a group"
                )
            child_key = "::".join(parts[: i + 1])
            child = current.get(child_key)
            # Bare name without @format — find the child whose key
            # matches after stripping the @suffix (e.g. "group::task"
            # matches "group::task@mcqa").
            if child is None and "@" not in parts[i]:
                candidates = [
                    k for k in current.child_names if k.split("@", 1)[0] == child_key
                ]
                if len(candidates) == 1:
                    child = current.get(candidates[0])
                elif len(candidates) > 1:
                    raise KeyError(
                        f"Ambiguous path '{child_key}': matches multiple "
                        f"children: {candidates}"
                    )
            if child is None:
                raise KeyError(
                    f"'{child_key}' not found in group '{current.name}'. "
                    f"Available children: {current.child_names}"
                )
            current = child

        return current

    def _load_spec(
        self, spec: str | Mapping[str, Any], overrides: dict[str, Any] | None = None
    ) -> Task | Group | list[Task]:
        """Load a task/group/tag by name, file path, or inline config.

        Args:
            spec: Task name (str), YAML file path, ``::``-separated path
                  (e.g. ``"group::subgroup::task"``), or dict with
                  "task"/"group" key.
            overrides: Runtime config overrides (e.g. ``num_fewshot``).
                  Should **not** contain ``task``/``group`` identification keys.

        Returns:
            Task, Group, or list[Task] (for tags)
        """
        overrides = overrides or {}
        match spec:
            # Registered name (possibly with @format selector)
            case str():
                # Try exact match first (handles "arc_easy@mcqa" registered names)
                entry = self._entry(spec)
                if entry:
                    return self._factory.build(
                        entry, overrides=overrides, registry=self._index
                    )

                # Handle :: path for inline groups/tasks
                if "::" in spec:
                    return self._resolve_path(spec, overrides=overrides)

                # If spec has @, try the base name with format as a runtime override
                if "@" in spec:
                    base_name, format_selection = spec.rsplit("@", 1)
                    fmt_overrides = {
                        "_formats_selection": format_selection,
                        **overrides,
                    }
                    entry = self._entry(base_name)
                    if entry:
                        return self._factory.build(
                            entry, overrides=fmt_overrides, registry=self._index
                        )

                # check if it's a path
                entry = TaskIndex.entry_from_path(Path(spec))
                if entry:
                    return self._factory.build(
                        entry, overrides=overrides, registry=self._index
                    )
                raise KeyError(
                    f"Spec '{spec}' is not a registered task/group/tag name or valid YAML path"
                )

            case dict():
                _entry = TaskIndex.entry_from_config(spec)
                if _entry:
                    # Identity keys are already captured in the Entry;
                    # pass only actual config overrides to the factory.
                    config_overrides = {
                        k: v for k, v in spec.items() if k not in ("task", "group")
                    }
                    return self._factory.build(
                        _entry,
                        overrides={**config_overrides, **overrides},
                        registry=self._index,
                    )
                raise ValueError(
                    "Invalid config dict: must contain 'task' or 'group' key"
                )
            case _:
                raise TypeError("spec must be str or dict")

    def load(
        self,
        specs: Sequence[str | Mapping[str, Any]],
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> TaskDict:
        """Resolve task specs into concrete [Task][lm_eval.api.task.Task] and [Group][lm_eval.api.group.Group] objects.

        Accepts name strings, config dicts, or a mix. Groups and
        tags references are expanded into their leaf tasks.

        Example:
            ```python
            loaded = task_manager.load(
                ["mmlu", "arc_easy"],
                overrides={
                    "arc_easy": {"num_fewshot": 5},
                    "mmlu": {"num_fewshot": 3},
                },
            )
            loaded["tasks"]  # {"mmlu_..": Task, ... "arc_easy": Task, ...}
            loaded["groups"]  # {"mmlu": Group}
            ```

        Args:
            specs: One or more task specs — a name string or a full config
                dict (e.g. ``{"task": "arc_easy", "doc_to_text": ..., ...}``).
            overrides: Optional mapping of task/group name to config overrides
                (e.g. ``{"arc_easy": {"num_fewshot": 5}}``). Only applied to
                string specs; dict specs carry their overrides inline.

        Returns:
            A [TaskDict][TaskDict] containing the loaded ``"tasks"``, ``"groups"``,
            and a ``"group_map"`` of each group to its immediate children.
        """
        if not isinstance(specs, list):
            specs = [specs]  # type: ignore
        _overrides = cast("dict[str, dict[str, Any]]", deepcopy(overrides or {}))

        # Build all requested items
        built: list[Task | Group] = []
        for spec in cast("Iterable", specs):
            # Dict specs are self-contained — they carry overrides inline
            # String specs look up by name in the overrides mapping
            spec_overrides = {} if isinstance(spec, dict) else _overrides.pop(spec, {})

            obj = self._load_spec(spec, overrides=spec_overrides)
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
                    tasks[task._qualified_name] = task
                for subgroup in item.get_all_groups():
                    groups[subgroup.name] = subgroup
            else:
                tasks[item._qualified_name] = item

        for item in built:
            collect(item)

        if _overrides:
            eval_logger.warning(
                "Unused overrides (no matching spec): %s",
                ", ".join(sorted(_overrides)),
            )

        return {
            "tasks": tasks,
            "groups": groups,
            "group_map": {g.name: g.child_names for g in groups.values()}
            if groups
            else {},
        }

    @deprecated("Use TaskManager.load(), which returns flat dicts of tasks and groups.")
    def load_task_or_group(self, task_list: str | list[str | dict]) -> dict:
        """Deprecated — use [load][.load] instead.

        Returns the old nested-dict format where groups are keyed by
        [ConfigurableGroup][lm_eval.api.group.ConfigurableGroup] and
        standalone tasks by name.

        Args:
            task_list: Task name(s) or override dicts.

        Returns:
            Nested dict — groups are keyed by ``ConfigurableGroup`` objects, standalone
                tasks by name. Subgroups recurse, e.g.
                ``{CG: {sub_CG: {task: Task, ...}, task: Task, ...}, ...}``.
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

    # ---------------------------------------------------------------- utility
    def match_tasks(self, task_list: list[str]) -> list[str]:
        """Match task names using glob patterns.

        Handles task@format syntax: strips @format for matching,
        returns the original (with @format) so _load_spec can parse it.
        """
        results = []
        for pattern in task_list:
            if "@" in pattern:
                base, preset_suffix = pattern.split("@", 1)
                matched = utils.pattern_match([base], self.all_tasks)
                results.extend(f"{m}@{preset_suffix}" for m in matched)
            else:
                matched = utils.pattern_match([pattern], self.all_tasks)
                results.extend(matched)
            if not matched:
                results.append(pattern)
        return sorted(set(results))

    def list_all_tasks(
        self,
        list_groups: bool = True,
        list_tags: bool = True,
        list_subtasks: bool = True,
    ) -> str:
        """Generate a Markdown table listing all available tasks."""
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
