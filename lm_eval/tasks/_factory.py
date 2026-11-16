from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from dataclasses import fields
from typing import TYPE_CHECKING, Any, cast

from lm_eval.api.group import Group
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.config.group import GroupConfig
from lm_eval.tasks._index import Entry, Kind
from lm_eval.tasks._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Mapping

eval_logger = logging.getLogger(__name__)

GROUP_FIELD_NAMES = {f.name for f in fields(GroupConfig)}


# TODO: DO not initialize Tasks, initialize in Manager
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
            case Entry(kind=Kind.TAG):
                return self._build_tag(entry, overrides, registry)

            case Entry(kind=Kind.GROUP):
                raw_cfg = self._load_full_config(entry, None)
                return self._build_group(entry.name, raw_cfg, overrides, registry)

            case _:
                return self._build_task(entry, overrides)

    # ---------------------------------------------------------------- build methods

    def _build_task(self, entry: Entry, overrides: dict[str, Any] | None) -> Task:
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
            obj = ConfigurableTask(config=cfg)

        return obj

    def _build_group(
        self,
        group_name: str,
        raw_cfg: dict[str, Any],
        overrides: dict[str, Any] | None,
        registry: dict[str, Entry],
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

        # Build Group object via existing from_config
        group = Group.from_config(group_cfg)

        # Task-level overrides = everything NOT a group structural key
        group_task_overrides = {
            k: v for k, v in raw_cfg.items() if k not in GROUP_FIELD_NAMES
        }
        # Merge: group-level defaults < caller overrides (caller takes precedence)
        merged_overrides = {**group_task_overrides, **(overrides or {})}

        # Build children from task: list (references to existing tasks/groups/tags)
        task_field = raw_cfg.get("task")
        if isinstance(task_field, list):
            for child in self._build_group_members(
                task_field, group_name, merged_overrides, registry
            ):
                group.add(child)

        return group

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
                            self._build_group(name, item, overrides, registry)
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
                    for task_name in sorted(child_entry.tags):
                        # TODO not adding namespaces currently
                        # namespaced = f"{group_name}::{task_name}"
                        namespaced = task_name
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
                    # TODO not adding namespaces currently
                    # namespaced = f"{group_name}::{base_name}"
                    namespaced = base_name
                    child_obj = self.build(
                        child_entry,
                        overrides={"task": namespaced, **item_overrides},
                        registry=registry,
                    )
                    children.append(cast("Task", child_obj))
                    registry[namespaced] = child_entry

        return children

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
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if entry.yaml_path:
            cfg = deepcopy(load_yaml(entry.yaml_path, resolve_func=True))
        elif entry.cfg:
            cfg = deepcopy(entry.cfg)
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
