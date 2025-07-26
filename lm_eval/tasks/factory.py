from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from lm_eval.api.group import GroupConfig
from lm_eval.api.task import ConfigurableTask, Task  # noqa: F401  (typing)
from lm_eval.tasks._config_loader import load_yaml as load_cfg
from lm_eval.tasks.index import Entry, Kind


load_cfg_cached = load_cfg  # type: ignore[no-redef]


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
        • entry.kind == TASK / PY_TASK  ➜ returns instantiated task object
        • entry.kind == GROUP          ➜ returns (GroupConfig, mapping-of-subtasks)
        • entry.kind == TAG            ➜ returns mapping-of-tasks (tag expansion)
        """
        if entry.kind is Kind.TAG:
            return self._build_tag(entry, overrides, registry)

        if entry.kind is Kind.GROUP:
            return self._build_group(entry, overrides, registry)

        return self._build_task(entry, overrides)

    def _build_task(self, entry: Entry, overrides: dict[str, Any] | None):
        cfg = self._load_full_config(entry, overrides)

        if "class" in cfg:  # PY_TASK route
            cls = cfg["class"]
            obj = cls(config=cfg) if _ctor_accepts_config(cls) else cls()
            if isinstance(obj, ConfigurableTask):
                obj.config.task = entry.name
            return obj

        # YAML task
        return ConfigurableTask(config=cfg)  # type: ignore[arg-type]

    def _build_group(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ):
        raw_cfg = self._load_full_config(entry, None)
        grp_cfg = {k: v for k, v in raw_cfg.items() if k in GroupConfig.__annotations__}
        grp_cfg["metadata"] = grp_cfg.get("metadata", {}) | self._meta
        group_obj = GroupConfig(**grp_cfg)

        children: dict[str, Any] = {}
        for item in group_obj.task:
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

            # `child` itself is a mapping (task-name -> obj) or {GroupConfig: ...}
            children.update(child)
        return {group_obj: children}

    def _build_tag(
        self,
        entry: Entry,
        overrides: dict[str, Any] | None,
        registry: Mapping[str, Entry],
    ):
        return {
            name: self._build_task(registry[name], overrides) for name in entry.tags
        }

    def _load_full_config(
        self, entry: Entry, overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        if entry.yaml_path:
            cfg = deepcopy(load_cfg_cached(entry.yaml_path, resolve_functions=True))
        else:
            cfg = {"metadata": {"config": "unknown"}}  # python task without YAML

        if overrides:
            cfg = {**cfg, **overrides}
        cfg["metadata"] = (
            m if isinstance(m := cfg.get("metadata", {}), dict) else {"_metadata": m}
        ) | self._meta
        cfg.setdefault("task", entry.name)
        return cfg


def _ctor_accepts_config(cls) -> bool:
    init = getattr(cls, "__init__", None)
    return init and "config" in inspect.signature(init).parameters
