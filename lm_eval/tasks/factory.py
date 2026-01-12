from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from lm_eval.api.group import AggMetricConfig, Group
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.tasks._config_loader import load_yaml
from lm_eval.tasks.index import Entry, Kind


if TYPE_CHECKING:
    from collections.abc import Mapping

eval_logger = logging.getLogger(__name__)


class TaskFactory:
    """
    Builds Task and Group objects from Entry definitions.

    Returns Task | Group directly - Groups contain their children.
    """

    def __init__(self, *, meta: dict[str, Any] | None = None):
        self._meta: dict[str, Any] = meta or {}

    # ---------------------------------------------------------------- public API

    def build(
        self,
        entry: Entry,
        *,
        overrides: dict[str, Any] | None = None,
        registry: dict[str, Entry],
    ) -> Task | Group | list[Task]:
        """
        Build an entry into a Task, Group, or list of Tasks.

        Returns:
            - Task: for single tasks
            - Group: for groups (with children populated)
            - list[Task]: for tags (expanded to multiple tasks)
        """
        match entry:
            # Handle external references (ref: in children)
            case Entry(ref_task=str() as ref):
                if ref not in registry:
                    raise KeyError(f"Reference '{ref}' not found for '{entry.name}'")
                # Use namespaced name for the task
                ref_overrides = {**(overrides or {}), "task": entry.name}
                return self.build(
                    registry[ref], overrides=ref_overrides, registry=registry
                )

            # Handle tag expansion (tag: in children)
            case Entry(tag_ref=str() as tag):
                if tag not in registry:
                    raise KeyError(f"Tag '{tag}' not found for '{entry.name}'")
                return self._build_tag(registry[tag], overrides, registry)

            case Entry(kind=Kind.TAG):
                return self._build_tag(entry, overrides, registry)

            case Entry(kind=Kind.GROUP):
                return self._build_group(entry, overrides, registry)

            case _:
                return self._build_task(entry, overrides)

    # ---------------------------------------------------------------- build methods

    def _build_task(self, entry: Entry, overrides: dict[str, Any] | None) -> Task:
        """Build and return a Task."""
        cfg = self._load_full_config(entry, overrides)

        # Remove structural keys that aren't part of task config
        for key in ("children", "tag", "group"):
            cfg.pop(key, None)

        task_name = cfg["task"]

        if "class" in cfg:  # PY_TASK route
            cls = cfg["class"]
            obj = cls(config=cfg) if _ctor_accepts_config(cls) else cls()
            if hasattr(obj, "config") and hasattr(obj.config, "task"):
                obj.config.task = task_name
        else:
            obj = ConfigurableTask(config=cfg)

        return obj

    def _build_group(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: dict[str, Entry],
    ) -> Group:
        """Build a Group with its children populated."""
        raw_cfg = self._load_full_config(entry, None)
        group_name = entry.name

        # Parse aggregation config
        aggregation = None
        if agg_list := raw_cfg.get("aggregate_metric_list"):
            if isinstance(agg_list, dict):
                agg_list = [agg_list]
            aggregation = [
                AggMetricConfig(**item) if isinstance(item, dict) else item
                for item in agg_list
            ]

        # Create the Group object
        group = Group(
            name=group_name,
            alias=raw_cfg.get("group_alias"),
            aggregation=aggregation,
            metadata=raw_cfg.get("metadata", {}) | self._meta,
        )

        # Extract task-config fields from group config for inheritance
        # These are keys that define group structure, not task config
        GROUP_ONLY_KEYS = {
            "group",
            "group_alias",
            "children",
            "task",
            "aggregate_metric_list",
            "metadata",
            "tag",
        }
        group_task_overrides = {
            k: v for k, v in raw_cfg.items() if k not in GROUP_ONLY_KEYS
        }
        # Merge: group-level defaults < caller overrides (caller takes precedence)
        merged_overrides = {**group_task_overrides, **(overrides or {})}

        # Build and add children from children: dict
        if "children" in raw_cfg:
            for child in self._build_children(
                raw_cfg["children"], group_name, merged_overrides, registry
            ):
                group.add(child)

        # Build and add children from old-style task: list (references to existing tasks/groups/tags)
        task_field = raw_cfg.get("task")
        if isinstance(task_field, list):
            for child in self._build_group_members(
                task_field, group_name, merged_overrides, registry
            ):
                group.add(child)

        return group

    def _build_children(
        self,
        children_cfg: dict[str, Any],
        group_name: str,
        overrides: dict[str, Any] | None,
        registry: dict[str, Entry],
    ) -> list[Task | Group]:
        """Build children defined via children: dict.

        Supported child formats:
        - Task reference: {task: "name", ...overrides}
        - Subgroup: {group: "name", task: [...], children: {...}}

        Returns:
            List of Task | Group objects
        """
        children: list[Task | Group] = []

        for child_name, child_cfg in children_cfg.items():
            child_path = f"{group_name}::{child_name}"

            if child_path not in registry:
                eval_logger.warning(f"Child '{child_path}' not found in registry")
                continue

            child_entry = registry[child_path]

            # Extract child-level overrides (skip structural keys)
            child_overrides = {
                k: v
                for k, v in child_cfg.items()
                if k not in ("task", "group", "children", "tag")
            }

            # Merge: parent overrides < child overrides
            merged = {**(overrides or {}), **child_overrides}

            # For task references, use namespaced name
            if child_entry.ref_task:
                merged["task"] = child_path

            child_obj = self.build(child_entry, overrides=merged, registry=registry)
            if isinstance(child_obj, (Task, Group)):
                children.append(child_obj)
            else:
                children.extend(child_obj)

        return children

    def _build_group_members(
        self,
        member_list: list[str | dict[str, Any]],
        group_name: str,
        overrides: dict[str, Any] | None,
        registry: dict[str, Entry],
    ) -> list[Task | Group]:
        """Build group members from task: list syntax.

        Handles references to existing tasks, groups, tags, and inline group definitions.
        Each item can be:
        - str: name of existing task/group/tag
        - dict with 'task': task reference with overrides
        - dict with 'group': inline subgroup definition

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
                # Check for inline group definition (legacy syntax)
                # e.g., {group: "stem", task: [...], aggregate_metric_list: [...]}
                if "group" in item:
                    inline_group = self._build_inline_group(
                        item, group_name, overrides, registry
                    )
                    children.append(inline_group)
                    continue

                base_name = item["task"]
                item_overrides = {**overrides, **item} if overrides else item
            else:
                raise TypeError(
                    f"Unsupported sub-entry {item!r} in group '{group_name}'"
                )

            # Handle inline task (not in registry)
            if base_name not in registry:
                namespaced = f"{group_name}::{base_name}"
                task_cfg: dict[str, Any] = {**item_overrides, "task": namespaced}
                task_cfg["metadata"] = task_cfg.get("metadata", {}) | self._meta
                children.append(ConfigurableTask(config=task_cfg))
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
                    for task_name in child_entry.tags:
                        namespaced = f"{group_name}::{task_name}"
                        child_obj = self.build(
                            registry[task_name],
                            overrides={"task": namespaced, **item_overrides},
                            registry=registry,
                        )
                        children.append(cast("Task", child_obj))
                        # also register to index with new namespaced name
                        registry[namespaced] = registry[task_name]

                case _:
                    # TASK or PY_TASK
                    namespaced = f"{group_name}::{base_name}"
                    child_obj = self.build(
                        child_entry,
                        overrides={"task": namespaced, **item_overrides},
                        registry=registry,
                    )
                    children.append(cast("Task", child_obj))
                    registry[namespaced] = child_entry

        return children

    def _build_inline_group(
        self,
        group_cfg: dict[str, Any],
        parent_name: str,
        overrides: dict[str, Any] | None,
        registry: dict[str, Entry],
    ) -> Group:
        """Build an inline group definition from legacy syntax.

        Handles: {group: "name", task: [...], aggregate_metric_list: [...]}
        """
        group_name = group_cfg["group"]
        namespaced_name = f"{parent_name}::{group_name}"

        # Parse aggregation config
        aggregation = None
        if agg_list := group_cfg.get("aggregate_metric_list"):
            if isinstance(agg_list, dict):
                agg_list = [agg_list]
            aggregation = [
                AggMetricConfig(**item) if isinstance(item, dict) else item
                for item in agg_list
            ]

        # Create the Group object
        group = Group(
            name=namespaced_name,
            alias=group_cfg.get("group_alias"),
            aggregation=aggregation,
            metadata=group_cfg.get("metadata", {}) | self._meta,
        )

        # Build children from task: list
        task_field = group_cfg.get("task")
        if isinstance(task_field, list):
            for child in self._build_group_members(
                task_field, namespaced_name, overrides, registry
            ):
                group.add(child)

        return group

    def _build_tag(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ) -> list[Task]:
        """Build all tasks from a tag.

        Tags are just a shorthand for multiple tasks, not a container.
        Returns a list of Task objects.
        """
        return [self._build_task(registry[name], overrides) for name in entry.tags]

    def _load_full_config(
        self, entry: Entry, overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        # For inline children (have parent), use the stored cfg directly
        # instead of loading from YAML (which would load the parent's full config)
        if entry.parent and entry.cfg:
            cfg = deepcopy(entry.cfg)
        elif entry.yaml_path:
            cfg = deepcopy(load_yaml(entry.yaml_path, resolve_func=True))
        else:
            cfg: dict[str, Any] = {
                "metadata": {"config": "unknown"}
            }  # python task without YAML

        # Handle task_list configs - merge base config with per-task overrides
        if "task_list" in cfg:
            task_list = cfg.pop("task_list")
            # Find the entry for this task in task_list
            for item in task_list:
                if isinstance(item, dict) and item.get("task") == entry.name:
                    # Merge per-task overrides
                    cfg = {**cfg, **item}
                    break

        if overrides:
            cfg = {**cfg, **overrides}
        cfg["metadata"] = (
            m if isinstance(m := cfg.get("metadata", {}), dict) else {"_metadata": m}
        ) | self._meta  # type: ignore
        cfg["metadata"]["config_source"] = str(entry.yaml_path) or "inline"
        cfg.setdefault("task", entry.name)
        return cfg


def _ctor_accepts_config(cls) -> bool:
    init = getattr(cls, "__init__", None)
    return bool(init and "config" in inspect.signature(init).parameters)
