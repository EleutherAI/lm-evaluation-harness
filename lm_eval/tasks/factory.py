from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._config_loader import load_yaml
from lm_eval.tasks.index import Entry, Kind


class TaskFactory:
    """
    Turns a *Entry* (plus optional overrides) into a
    *Task* | *ConfigurableTask* | *GroupConfig* hierarchy.
    """

    def __init__(self, *, meta: dict[str, Any] | None = None):
        self._meta = meta or {}

    # ---------------------------------------------------------------- public API
    def build(
        self,
        entry: Entry,
        *,
        overrides: dict[str, Any] | None = None,
        registry: Mapping[str, Entry],
    ):
        """
        * entry.kind == TASK / PY_TASK -> returns instantiated task object
        * entry.kind == GROUP -> returns (GroupConfig, mapping-of-subtasks)
        * entry.kind == TAG -> returns mapping-of-tasks (tag expansion)
        * entry with ref_target -> resolves reference and builds target
        * entry with tag_ref -> expands tag and builds tasks
        """
        # Handle external references (ref: in children)
        if entry.ref_target:
            if entry.ref_target not in registry:
                raise KeyError(
                    f"Reference '{entry.ref_target}' not found for '{entry.name}'"
                )
            target_entry = registry[entry.ref_target]
            return self.build(target_entry, overrides=overrides, registry=registry)

        # Handle tag expansion (tag: in children)
        if entry.tag_ref:
            if entry.tag_ref not in registry:
                raise KeyError(f"Tag '{entry.tag_ref}' not found for '{entry.name}'")
            tag_entry = registry[entry.tag_ref]
            return self._build_tag(tag_entry, overrides, registry)

        if entry.kind is Kind.TAG:
            return self._build_tag(entry, overrides, registry)

        if entry.kind is Kind.GROUP:
            return self._build_group(entry, overrides, registry)

        return self._build_task(entry, overrides)

    def _build_task(self, entry: Entry, overrides: dict[str, Any] | None):
        """Build a task and return it wrapped in a dict {task_name: task_obj}."""
        cfg = self._load_full_config(entry, overrides)

        # Remove structural keys that aren't part of task config
        for key in ("children", "ref", "tag", "group"):
            cfg.pop(key, None)

        # Use cfg["task"] as key (may be overridden, e.g., for namespacing)
        task_name = cfg["task"]

        if "class" in cfg:  # PY_TASK route
            cls = cfg["class"]
            obj = cls(config=cfg) if _ctor_accepts_config(cls) else cls()
            if hasattr(obj, "config") and hasattr(obj.config, "task"):
                obj.config.task = task_name
            return {task_name: obj}

        # Regular YAML task - use ConfigurableTask
        task_obj = ConfigurableTask(config=cfg)
        return {task_name: task_obj}

    def _build_group(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ):
        raw_cfg = self._load_full_config(entry, None)
        grp_cfg = {k: v for k, v in raw_cfg.items() if k in GroupConfig.__annotations__}
        grp_cfg["metadata"] = grp_cfg.get("metadata", {}) | self._meta
        group_obj = ConfigurableGroup(config=grp_cfg)
        group_name = entry.name

        children: dict[str, Any] = {}

        # Handle new-style children: dict (hierarchical)
        if "children" in raw_cfg:
            children.update(
                self._build_children(
                    raw_cfg["children"], group_name, overrides, registry
                )
            )

        # Handle old-style task: list (backward compatibility)
        if "task" in grp_cfg and isinstance(grp_cfg["task"], list):
            children.update(
                self._build_task_list(grp_cfg["task"], group_name, overrides, registry)
            )

        return {group_obj: children}

    def _build_children(
        self,
        children_cfg: dict[str, Any],
        group_name: str,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ) -> dict[str, Any]:
        """Build children defined via children: dict."""
        result: dict[str, Any] = {}

        for child_name, child_cfg in children_cfg.items():
            child_path = f"{group_name}::{child_name}"

            # Look up pre-registered entry from index
            if child_path in registry:
                child_entry = registry[child_path]
                child_overrides = overrides or {}

                # Merge any inline overrides from child_cfg (excluding structural keys)
                inline_overrides = {
                    k: v
                    for k, v in child_cfg.items()
                    if k not in ("ref", "tag", "children")
                }
                if inline_overrides:
                    child_overrides = {**child_overrides, **inline_overrides}

                child = self.build(
                    child_entry, overrides=child_overrides, registry=registry
                )
                result.update(child)
            else:
                # Fallback: inline task not pre-registered (shouldn't normally happen)
                task_cfg = {**child_cfg, "task": child_path}
                task_cfg["metadata"] = task_cfg.get("metadata", {}) | self._meta
                result[child_path] = ConfigurableTask(config=task_cfg)

        return result

    def _build_task_list(
        self,
        task_list: list,
        group_name: str,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ) -> dict[str, Any]:
        """Build children defined via task: list (backward compatibility)."""
        result: dict[str, Any] = {}

        for item in task_list:
            # Step 1: Normalize - extract base_name and item_overrides
            if isinstance(item, str):
                base_name = item
                item_overrides = overrides or {}
            elif isinstance(item, dict):
                base_name = item["task"]
                item_overrides = {**overrides, **item}
            else:
                raise TypeError(
                    f"Unsupported sub-entry {item!r} in group '{group_name}'"
                )

            # Step 2: Handle inline task (not in registry)
            if base_name not in registry:
                namespaced = f"{group_name}::{base_name}"
                task_cfg = {**item_overrides, "task": namespaced}
                task_cfg["metadata"] = task_cfg.get("metadata", {}) | self._meta
                result[namespaced] = ConfigurableTask(config=task_cfg)
                continue

            # Step 3: Build based on entry kind
            child_entry = registry[base_name]

            if child_entry.kind is Kind.GROUP:
                child = self.build(
                    child_entry, overrides=item_overrides, registry=registry
                )
            elif child_entry.kind is Kind.TAG:
                child = {}
                for task_name in child_entry.tags:
                    namespaced = f"{group_name}::{task_name}"
                    child.update(
                        self.build(
                            registry[task_name],
                            overrides={"task": namespaced, **item_overrides},
                            registry=registry,
                        )
                    )
            else:  # TASK or PY_TASK
                namespaced = f"{group_name}::{base_name}"
                child = self.build(
                    child_entry,
                    overrides={"task": namespaced, **item_overrides},
                    registry=registry,
                )

            result.update(child)

        return result

    def _build_tag(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ):
        """Build all tasks in a tag and return merged dict."""
        result = {}
        for name in entry.tags:
            result.update(self._build_task(registry[name], overrides))
        return result

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
        ) | self._meta
        cfg.setdefault("task", entry.name)
        return cfg


def _ctor_accepts_config(cls) -> bool:
    init = getattr(cls, "__init__", None)
    return bool(init and "config" in inspect.signature(init).parameters)
