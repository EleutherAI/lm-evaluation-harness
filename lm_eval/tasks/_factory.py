from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

from lm_eval.api.group import Group
from lm_eval.api.task import ConfigurableTask, Task
from lm_eval.config.group import GroupConfig
from lm_eval.config.task import TaskConfig
from lm_eval.tasks._index import Entry, Kind
from lm_eval.tasks._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Mapping

    from lm_eval.api.instance import OutputType

eval_logger = logging.getLogger(__name__)

GROUP_FIELD_NAMES = {f.name for f in fields(GroupConfig)}


# TODO: DO not initialize Tasks, initialize in Manager
class TaskFactory:
    """
    Builds Task and Group objects from Entry definitions.

    Returns Task | Group directly - Groups contain their children.
    """

    def __init__(
        self,
        *,
        meta: dict[str, Any] | None = None,
        output_type: OutputType | None = None,
    ):
        self.output_type = output_type
        self._skipped = 0
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
                item = self._build_group(entry.name, raw_cfg, overrides, registry)

            case _:
                item = self._build_task(entry, overrides)
        return item if item is not None else []

    # ---------------------------------------------------------------- build methods

    def _build_task(
        self, entry: Entry, overrides: dict[str, Any] | None
    ) -> Task | None:
        """Skip incompatible configurable tasks before loading functions or datasets."""
        if self.output_type and entry.kind == Kind.TASK:
            cfg = (entry.cfg or {}) | (overrides or {})
            if not self._matches_type(cfg.get("output_type", TaskConfig.output_type)):
                return None
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
            return ConfigurableTask(config=cfg)

        # Custom Python tasks can set their request type in the constructor.
        return (
            obj if not self.output_type or self._matches_type(obj.OUTPUT_TYPE) else None
        )

    def select_initialized(self, item: Task | Group) -> Task | Group | None:
        """Filter caller-provided objects without mutating their group membership."""
        if not self.output_type:
            return item
        if isinstance(item, Group):
            skipped_before = self._skipped
            group = replace(item, _children={})
            for child in item:
                selected = self.select_initialized(child)
                if selected is not None:
                    group.add(selected)
            return self._finish_group(group, skipped_before)
        return item if self._matches_type(item.OUTPUT_TYPE) else None

    def _matches_type(self, task_type: OutputType | None) -> bool:
        if self.output_type is None or self.output_type == task_type:
            return True
        self._skipped += 1
        return False

    def _finish_group(self, group: Group, skipped_before: int) -> Group | None:
        skipped = self._skipped - skipped_before
        if skipped:
            selected = len(group.get_all_tasks())
            eval_logger.info(
                "Group %s: selected %d/%d tasks for %s; omitting group score.",
                group.name,
                selected,
                selected + skipped,
                self.output_type,
            )
            group.aggregate_metric_list = None
            if not selected:
                return None
        return group

    def _build_group(
        self,
        group_name: str,
        raw_cfg: dict[str, Any],
        overrides: dict[str, Any] | None,
        registry: dict[str, Entry],
    ) -> Group | None:
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
        skipped_before = self._skipped
        task_field = raw_cfg.get("task")
        if isinstance(task_field, list):
            for child in self._build_group_members(
                task_field, group_name, merged_overrides, registry
            ):
                group.add(child)

        return self._finish_group(group, skipped_before)

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
                        subgroup = self._build_group(
                            name,
                            self._load_full_config(registry[name]),
                            merged,
                            registry,
                        )
                    else:
                        name = f"{group_name}::{name}"
                        subgroup = self._build_group(name, item, overrides, registry)
                    if subgroup is not None:
                        children.append(subgroup)
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
                if not self._matches_type(
                    task_cfg.get("output_type", TaskConfig.output_type)
                ):
                    continue
                children.append(ConfigurableTask(config=task_cfg))
                continue

            child = self.build(
                registry[base_name], overrides=item_overrides, registry=registry
            )
            if isinstance(child, list):
                children.extend(child)
            else:
                children.append(child)

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
            task = self._build_task(registry[name], {**(overrides or {}), "task": name})
            if task is not None:
                tasks.append(task)
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
