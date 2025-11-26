from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from lm_eval.api.group import ConfigurableGroup, GroupConfig
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._config_loader import load_yaml as load_cfg
from lm_eval.tasks.index import Entry, Kind


load_cfg_cached = load_cfg  # type: ignore[no-redef]


class TaskFactory:
    """
    Turns a *Entry* (plus optional overrides) into a
    *Task* (from task_v3) | *ConfigurableTask* | *GroupConfig* hierarchy.

    For YAML tasks, uses the task_v3.Task builder pattern to automatically
    select the appropriate Task subclass based on output_type.
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
        • entry.kind == TASK / PY_TASK ➜ returns instantiated task object
        • entry.kind == GROUP ➜ returns (GroupConfig, mapping-of-subtasks)
        • entry.kind == TAG ➜ returns mapping-of-tasks (tag expansion)
        """
        if entry.kind is Kind.TAG:
            return self._build_tag(entry, overrides, registry)

        if entry.kind is Kind.GROUP:
            return self._build_group(entry, overrides, registry)

        return self._build_task(entry, overrides)

    def _build_task(self, entry: Entry, overrides: dict[str, Any] | None) -> dict:
        """Build a task and return it wrapped in a dict {task_name: task_obj}."""
        cfg = self._load_full_config(entry, overrides)

        if "class" in cfg:  # PY_TASK route
            cls = cfg["class"]
            obj = cls(config=cfg) if _ctor_accepts_config(cls) else cls()
            if hasattr(obj, "config") and hasattr(obj.config, "task"):
                obj.config.task = entry.name
            return {entry.name: obj}

        # Regular YAML task - use ConfigurableTask
        task_obj = ConfigurableTask(config=cfg)
        return {entry.name: task_obj}

    def _build_group(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ):
        raw_cfg = self._load_full_config(entry, None)
        grp_cfg = {k: v for k, v in raw_cfg.items() if k in GroupConfig.__annotations__}
        grp_cfg["metadata"] = grp_cfg.get("metadata", {}) | self._meta
        # Use ConfigurableGroup (hashable) instead of GroupConfig (dict, unhashable)
        group_obj = ConfigurableGroup(config=grp_cfg)

        children: dict[str, Any] = {}
        for item in group_obj.config["task"]:
            if isinstance(item, str):  # task: hellaswag
                child = self.build(
                    registry[item],
                    overrides=overrides,  # group-level overrides propagate
                    registry=registry,
                )
            elif isinstance(item, dict):  # task: {task: hellaswag, num_fewshot: 5}
                base_name = item["task"]
                child = self.build(
                    registry[base_name],
                    overrides=item,  # per-item override
                    registry=registry,
                )
            else:
                raise TypeError(
                    f"Unsupported sub-entry {item!r} in group '{entry.name}'"
                )

            # `child` itself is a mapping (task-name -> obj) or {ConfigurableGroup: ...}
            children.update(child)
        return {group_obj: children}

    def _build_tag(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ) -> dict:
        """Build all tasks in a tag and return merged dict."""
        result = {}
        for name in entry.tags:
            result.update(self._build_task(registry[name], overrides))
        return result

    def _load_full_config(
        self, entry: Entry, overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        if entry.yaml_path:
            cfg = deepcopy(load_cfg_cached(entry.yaml_path, resolve_func=True))
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
