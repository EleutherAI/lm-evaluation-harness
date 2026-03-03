from __future__ import annotations

import inspect
import logging
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lm_eval.api.group import Group
from lm_eval.api.task import Task
from lm_eval.config.group import GroupConfig

from ._index import Entry, Kind
from ._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Mapping

eval_logger = logging.getLogger(__name__)

GROUP_FIELD_NAMES = {f.name for f in fields(GroupConfig)}


# TODO: DO not initialize Tasks, initialize in Manager
class TaskFactory:
    """Builds Task and Group objects from Entry definitions.

    Returns Task | Group directly - Groups contain references to their children.
    """

    def __init__(self, *, meta: dict[str, Any] | None = None):
        self._meta: dict[str, Any] = meta or {}

    def _merge_metadata(self, existing: Any, config_source: str) -> dict[str, Any]:
        """Merge *existing* metadata with runtime metadata (``self._meta``).

        Handles ``None``, missing, and non-dict metadata values.
        Always sets ``config_source`` on the result. joins if existing.
        """
        if existing is None:
            metadata: dict[str, Any] = {}
        elif not isinstance(existing, dict):
            metadata = {"_metadata": existing}
        else:
            metadata = existing
        _config_source = str(metadata.get("config_source", "")) + f",{config_source}"
        return {**metadata, **self._meta, "config_source": config_source}

    # ---------------------------------------------------------------- public API

    def build(
        self,
        entry: Entry,
        *,
        overrides: Mapping[str, Any] | None = None,
        registry: dict[str, Entry],
    ) -> Task | Group | list[Task]:
        """Build an entry into a Task, Group, or list of Tasks.

        Returns:
            - Task: for single tasks
            - Group: for groups (with children populated)
            - list[Task]: for tags (expanded to multiple tasks)
        """
        match entry:
            case Entry(kind=Kind.TAG):
                return self._build_tag(entry, overrides, registry)

            case Entry(kind=Kind.GROUP):
                raw_cfg = self._load_full_config(entry, None)
                return self._build_group(
                    entry.name, raw_cfg, overrides, registry, entry.yaml_path
                )

            case _:
                return self._build_task(entry, overrides)

    # ---------------------------------------------------------------- build methods

    def _build_task(self, entry: Entry, overrides: Mapping[str, Any] | None) -> Task:
        """Build and return a Task."""
        cfg = self._load_full_config(entry, overrides)

        # Remove YAML indexing fields that aren't part of task config
        for key in ("tag", "task_list"):
            cfg.pop(key, None)

        task_name = cfg["task"]

        if "class" in cfg:  # PY_TASK route
            cls = cfg["class"]
            obj = cls(config=cfg) if _ctor_accepts_config(cls) else cls()
            if hasattr(obj, "config") and hasattr(obj.config, "task"):
                obj.config.task = task_name
        else:
            obj = Task.from_config(cfg)

        return obj

    def _build_group(
        self,
        group_name: str,
        raw_cfg: Mapping[str, Any],
        overrides: Mapping[str, Any] | None,
        registry: dict[str, Entry],
        yaml_path: Path | None = None,
    ) -> Group:
        """Build a Group with its children populated.

        Works for both registry-based groups (Entry with yaml) and
        inline group dicts defined inside a parent's task: list.
        """
        # Separate group-config fields from task-override fields
        group_dict = {k: v for k, v in raw_cfg.items() if k in GROUP_FIELD_NAMES}
        group_dict["group"] = group_name

        # Parse through GroupConfig (normalizes aggregate_metric_list, task, etc.)
        group_cfg = GroupConfig(**group_dict)

        group_cfg.metadata = self._merge_metadata(
            group_cfg.metadata, str(yaml_path) if yaml_path else "inline"
        )

        # Build Group object via existing from_config
        group = Group.from_config(group_cfg)

        merged_overrides = self._compute_child_overrides(
            overrides, group_cfg.include, yaml_path
        )

        # Build children from task: list (references to existing tasks/groups/tags)
        task_field = raw_cfg.get("task")
        if isinstance(task_field, list):
            for child in self._build_group_members(
                task_field, group_name, merged_overrides, registry, yaml_path
            ):
                group.add(child)

        return group

    def _compute_child_overrides(
        self,
        caller_overrides: Mapping[str, Any] | None,
        include: str | Mapping[str, Any] | None,
        yaml_path: Path | None,
    ) -> dict[str, Any]:
        """Compute the effective overrides that will be passed to child tasks.

        Merge precedence (lowest → highest):
            1. **Include defaults** — loaded from the ``include:`` field
               (a YAML path or inline dict on the group config).
            2. **Caller overrides** — runtime overrides passed by the user
               (e.g. via ``load(specs, overrides={...})``) with structural
               group keys (``group``, ``task``, ``include``, …) stripped so
               they don't leak into child task configs.
        """
        include_defaults = self._resolve_include(include, yaml_path)

        caller_task_overrides = {
            k: v
            for k, v in (caller_overrides or {}).items()
            if k not in GROUP_FIELD_NAMES
        }

        return {**include_defaults, **caller_task_overrides}

    def _build_group_members(
        self,
        member_list: list[str | dict[str, Any]],
        group_name: str,
        overrides: Mapping[str, Any] | None,
        registry: dict[str, Entry],
        yaml_path: Path | None = None,
    ) -> list[Task | Group]:
        """Build group members from the ``task:`` list in a group config.

        Each item can be a string name, a ``{task: ...}`` dict, or a
        ``{group: ...}`` dict.  See the individual ``_build_*`` helpers
        for how each form is handled.
        """
        children: list[Task | Group] = []
        for item in member_list:
            if isinstance(item, dict) and "group" in item:
                children.append(
                    self._build_group_ref(
                        item, group_name, overrides, registry, yaml_path
                    )
                )
            else:
                base_name, effective = self._resolve_member(item, overrides, group_name)
                children.extend(
                    self._build_by_kind(
                        base_name, effective, group_name, registry, yaml_path
                    )
                )
        return children

    # -- member helpers (called from _build_group_members) -----------------

    @staticmethod
    def _resolve_member(
        item: str | dict[str, Any],
        overrides: Mapping[str, Any] | None,
        group_name: str,
    ) -> tuple[str, Mapping[str, Any]]:
        """Normalize a member-list item into ``(base_name, effective_overrides)``.

        Handles two forms:
        - **str**: bare task/group/tag name — inherits parent overrides as-is.
        - **dict with 'task'**: task reference with per-item overrides merged
          on top of the parent overrides.
        """
        if isinstance(item, str):
            return item, overrides or {}

        if isinstance(item, dict):
            if "task" not in item:
                raise ValueError(
                    f"Dict entry in group '{group_name}' must have 'task' or "
                    f"'group' key, got: {list(item.keys())}"
                )
            base_name = item["task"]
            effective = {**overrides, **item} if overrides else item
            return base_name, effective

        raise TypeError(f"Unsupported sub-entry {item!r} in group '{group_name}'")

    def _build_group_ref(
        self,
        item: dict[str, Any],
        group_name: str,
        overrides: Mapping[str, Any] | None,
        registry: dict[str, Entry],
        yaml_path: Path | None,
    ) -> Group:
        """Build a ``{group: ...}`` member-list entry.

        Two sub-cases:
        - **Name in registry**: reference to an existing group — merge the
          dict's extra keys as overrides on top of the parent overrides.
        - **Name not in registry**: true inline group definition — namespace
          it under the parent and build from the raw dict.
        """
        name = item["group"]
        if name in registry:
            inline_overrides = {k: v for k, v in item.items() if k != "group"}
            merged = {**(overrides or {}), **inline_overrides}
            return cast(
                "Group", self.build(registry[name], overrides=merged, registry=registry)
            )

        # Inline group definition (not registered)
        namespaced = f"{group_name}::{name}"
        return self._build_group(namespaced, item, overrides, registry, yaml_path)

    def _build_by_kind(
        self,
        base_name: str,
        overrides: Mapping[str, Any],
        group_name: str,
        registry: dict[str, Entry],
        yaml_path: Path | None,
    ) -> list[Task | Group]:
        """Look up *base_name* in the registry and build by entry kind.

        Returns a list because TAG entries expand to multiple tasks.
        For TASK/PY_TASK/GROUP a single-element list is returned.

        If *base_name* is not in the registry it is treated as an inline
        task definition (namespaced under *group_name*).
        """
        namespaced = f"{group_name}::{base_name}"

        # Inline task — not in the registry
        if base_name not in registry:
            task_cfg: dict[str, Any] = {
                **overrides,
                "task": namespaced,
                "_qualified_name": namespaced,
            }
            task_cfg.setdefault("task_alias", base_name)
            task_cfg["metadata"] = self._merge_metadata(
                task_cfg.get("metadata"), str(yaml_path) if yaml_path else "inline"
            )
            return [Task.from_config(task_cfg)]

        entry = registry[base_name]

        if entry.kind is Kind.GROUP:
            return [
                cast("Group", self.build(entry, overrides=overrides, registry=registry))
            ]

        if entry.kind is Kind.TAG:
            tasks: list[Task | Group] = []
            for task_name in sorted(entry.tags):
                ns = f"{group_name}::{task_name}"
                tasks.append(
                    cast(
                        "Task",
                        self.build(
                            registry[task_name],
                            overrides={"_qualified_name": ns, **overrides},
                            registry=registry,
                        ),
                    )
                )
            return tasks

        # TASK or PY_TASK
        return [
            cast(
                "Task",
                self.build(
                    entry,
                    overrides={"_qualified_name": namespaced, **overrides},
                    registry=registry,
                ),
            )
        ]

    @staticmethod
    def _resolve_include(
        include: str | Mapping[str, Any] | None,
        yaml_path: Path | None,
    ) -> dict[str, Any]:
        """Resolve a GroupConfig ``include`` value into task-override fields.

        Args:
            include: Path to a YAML file, an inline dict, or None.
            yaml_path: Path of the group YAML file (for relative path resolution).

        Returns:
            Dict of task-level override fields (empty if include is None).
        """
        match include:
            # fmt: off
            case None:
                return {}
            case dict():
                return {**include}
            # fmt: on
            case str():
                # String path — load YAML file
                inc = Path(include)
                if not inc.is_absolute():
                    if yaml_path is None:
                        raise ValueError(
                            f"Cannot resolve relative include path '{include}' "
                            "without a YAML file context (inline group config)."
                        )
                    inc = yaml_path.parent / inc
                return load_yaml(inc, resolve_func=True)
            case _:
                raise TypeError(f"Invalid include value: {include!r}")

    def _build_tag(
        self,
        entry: Entry,
        overrides: Mapping[str, Any] | None,
        registry: Mapping[str, Entry],
    ) -> list[Task]:
        """Build all tasks from a tag.

        Tags are just a shorthand for multiple tasks, not a container.
        Returns a list of Task objects.
        """
        tasks = []
        for name in sorted(entry.tags):
            if name not in registry:
                eval_logger.warning(
                    "Tag %r references unknown task %r, skipping.", entry.name, name
                )
                continue
            tasks.append(self._build_task(registry[name], overrides))
        return tasks

    def _load_full_config(
        self,
        entry: Entry,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if entry.yaml_path:
            cfg = load_yaml(entry.yaml_path, resolve_func=True)
        elif entry.cfg:
            cfg = {**entry.cfg}
        else:
            eval_logger.debug(
                "Entry %r has no YAML or inline config; using placeholder.", entry.name
            )
            cfg: dict[str, Any] = {
                "metadata": {"config": "unknown"}
            }  # python task without YAML

        if overrides:
            cfg = {**cfg, **overrides}
        cfg["metadata"] = self._merge_metadata(
            cfg.get("metadata"), str(entry.yaml_path) if entry.yaml_path else "inline"
        )
        return cfg


def _ctor_accepts_config(cls) -> bool:
    init = getattr(cls, "__init__", None)
    return bool(init and "config" in inspect.signature(init).parameters)
