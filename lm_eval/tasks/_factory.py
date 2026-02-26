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

        # Remove structural keys that aren't part of task config
        for key in ("tag", "group", "task_list"):
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

        # Merge runtime metadata
        group_cfg.metadata = (group_cfg.metadata or {}) | self._meta
        group_cfg.metadata.setdefault(
            "config_source", str(yaml_path) if yaml_path else "inline"
        )

        # Build Group object via existing from_config
        group = Group.from_config(group_cfg)

        # Resolve include → explicit task defaults
        include_overrides = self._resolve_include(group_cfg.include, yaml_path)

        # Implicit task-level overrides = everything NOT a group structural key
        # (backward compat for groups that put task fields at top level)
        group_task_overrides = {
            k: v for k, v in raw_cfg.items() if k not in GROUP_FIELD_NAMES
        }

        # Filter group structural keys from caller overrides so they don't
        # leak into child task configs (e.g. task: ["arc_easy"] from the
        # group spec should not clobber a child's task: "arc_easy" string).
        caller_task_overrides = {
            k: v for k, v in (overrides or {}).items() if k not in GROUP_FIELD_NAMES
        }

        # Merge: implicit top-level < include < caller overrides
        merged_overrides = {
            **group_task_overrides,
            **include_overrides,
            **caller_task_overrides,
        }

        # Build children from task: list (references to existing tasks/groups/tags)
        task_field = raw_cfg.get("task")
        if isinstance(task_field, list):
            for child in self._build_group_members(
                task_field, group_name, merged_overrides, registry, yaml_path
            ):
                group.add(child)

        return group

    def _build_group_members(
        self,
        member_list: list[str | dict[str, Any]],
        group_name: str,
        overrides: Mapping[str, Any] | None,
        registry: dict[str, Entry],
        yaml_path: Path | None = None,
    ) -> list[Task | Group]:
        """Build group members from task: list syntax.

        Handles references to existing tasks, groups, tags, and inline group definitions.
        Each item can be:
        - str: name of existing task/group/tag
        - dict with 'task': task reference with overrides
        - dict with 'group': checks registry, otherwise new inline subgroup definition

        Returns:
            List of Task | Group objects
        """
        children: list[Task | Group] = []

        for item in member_list:
            # Normalize - extract base_name and item_overrides
            if isinstance(item, str):
                base_name = item
                item_overrides = overrides or {}
            elif isinstance(item, dict):
                # Dict with 'group' key — either a reference to an
                # existing group (with overrides) or a true inline definition.
                if "group" in item:
                    name = item["group"]
                    if name in registry:
                        # Existing group referenced with overrides
                        inline_overrides = {
                            k: v for k, v in item.items() if k != "group"
                        }
                        merged = {**(overrides or {}), **inline_overrides}
                        child_obj = self.build(
                            registry[name],
                            overrides=merged,
                            registry=registry,
                        )
                        children.append(cast("Group", child_obj))
                    else:
                        # True inline group (not in registry)
                        name = f"{group_name}::{name}"
                        children.append(
                            self._build_group(
                                name, item, overrides, registry, yaml_path
                            )
                        )
                    continue

                if "task" not in item:
                    raise ValueError(
                        f"Dict entry in group '{group_name}' must have 'task' or 'group' key, got: {list(item.keys())}"
                    )
                base_name = item["task"]
                item_overrides = {**overrides, **item} if overrides else item
            else:
                raise TypeError(
                    f"Unsupported sub-entry {item!r} in group '{group_name}'"
                )

            # Handle inline task (not in registry)
            if base_name not in registry:
                namespaced = f"{group_name}::{base_name}"
                task_cfg: dict[str, Any] = {
                    **item_overrides,
                    "task": namespaced,
                    "_qualified_name": namespaced,
                }
                task_cfg.setdefault("task_alias", base_name)
                task_cfg["metadata"] = task_cfg.get("metadata", {}) | self._meta
                task_cfg["metadata"]["config_source"] = (
                    str(yaml_path) if yaml_path else "inline"
                )
                children.append(Task.from_config(task_cfg))
                continue

            # Build based on entry kind
            child_entry = registry[base_name]
            match child_entry:
                case Entry(kind=Kind.GROUP):
                    child_obj = self.build(
                        child_entry, overrides=item_overrides, registry=registry
                    )
                    children.append(cast("Group", child_obj))

                case Entry(kind=Kind.TAG):
                    for task_name in sorted(child_entry.tags):
                        namespaced = f"{group_name}::{task_name}"
                        child_obj = self.build(
                            registry[task_name],
                            overrides={"_qualified_name": namespaced, **item_overrides},
                            registry=registry,
                        )
                        children.append(cast("Task", child_obj))
                        registry[namespaced] = registry[task_name]

                case _:
                    # TASK or PY_TASK
                    namespaced = f"{group_name}::{base_name}"
                    child_obj = self.build(
                        child_entry,
                        overrides={"_qualified_name": namespaced, **item_overrides},
                        registry=registry,
                    )
                    children.append(cast("Task", child_obj))
                    registry[namespaced] = child_entry

        return children

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
                    f"Tag '{entry.name}' references unknown task '{name}', skipping."
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
                f"Entry '{entry.name}' has no YAML or inline config; using placeholder."
            )
            cfg: dict[str, Any] = {
                "metadata": {"config": "unknown"}
            }  # python task without YAML

        if overrides:
            cfg = {**cfg, **overrides}
        cfg["metadata"] = (
            m if isinstance(m := cfg.get("metadata", {}), dict) else {"_metadata": m}
        ) | self._meta  # type: ignore
        cfg["metadata"]["config_source"] = (
            str(entry.yaml_path) if entry.yaml_path else "inline"
        )
        return cfg


def _ctor_accepts_config(cls) -> bool:
    init = getattr(cls, "__init__", None)
    return bool(init and "config" in inspect.signature(init).parameters)
