from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from lm_eval.tasks._config_loader import load_yaml as load_cfg


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class Kind(Enum):
    TASK = auto()  # YAML task, or task_list entry
    PY_TASK = auto()  # Python-defined, via "class"
    GROUP = auto()
    TAG = auto()
    TASK_LIST = auto()


@dataclass
class Entry:
    name: str
    kind: Kind
    yaml_path: Path | None  # None for generated / py-only entries
    cfg: dict[str, str] | None = None
    tags: set[str] = field(default_factory=set)
    task_list_path: Path | None = None


log = logging.getLogger(__name__)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


class TaskIndex:
    """Walks one or more directories, parses YAML quickly (functions unresolved),
    and produces a mapping {task_name: Entry}.
    """

    def __init__(self, *, meta: dict[str, str] | None = None) -> None:
        self._metadata = meta or {}

    def build(
        self,
        paths: Iterable[Path],
        *,
        resolve_includes=True,
    ) -> dict[str, Entry]:
        index: dict[str, Entry] = {}
        log.debug("Building task index from %s", paths)
        for root in paths:
            for yaml_path in self._iter_yaml_files(root):
                try:
                    cfg = load_cfg(
                        yaml_path,
                        resolve_func=False,
                        recursive=resolve_includes,
                    )
                    self.process_cfg(cfg, yaml_path, index)
                except Exception as err:
                    log.debug("Skip %s (%s)", yaml_path, err)
                    continue

                # self._process_cfg(cfg, yaml_path, index)
        log.debug("Built task index with %d entries", len(index))
        return index

    @staticmethod
    def _iter_yaml_files(root: Path):
        # Sort for deterministic traversal order across filesystems
        yield from sorted(
            (
                p
                for p in root.glob("**/*.yaml")
                if not any(part in _IGNORE_DIRS for part in p.parts)
            ),
            key=lambda p: p.as_posix(),
        )

    @staticmethod
    def process_cfg(
        cfg: dict[str, Any],
        path: Path,
        index: dict[str, Entry],
    ) -> None:
        kind = TaskIndex._kind_of(cfg)
        if kind is Kind.GROUP:
            grp_name = cfg["group"]
            if grp_name in index:
                log.debug(
                    f"Duplicate group name '{grp_name}' found. "
                    f"Already registered from: {index[grp_name].yaml_path}. "
                    f"Skipping duplicate from: {path}"
                )
                return
            index[grp_name] = Entry(
                name=grp_name,
                kind=Kind.GROUP,
                yaml_path=path,
                tags=TaskIndex._str_to_set(cfg.get("tag")),
                cfg=cfg,
            )
            return

        if kind is Kind.TASK or kind is Kind.PY_TASK:
            name = cfg["task"]
            if name in index:
                log.warning(
                    f"Duplicate task name '{name}' found. "
                    f"Already registered from: {index[name].yaml_path}. "
                    f"Skipping duplicate from: {path}"
                )
                return
            index[name] = Entry(
                name=name,
                kind=Kind.TASK,
                yaml_path=path,
                tags=TaskIndex._str_to_set(cfg.get("tag")),
                cfg=cfg,
            )
            TaskIndex._register_tags(name, cfg.get("tag"), index)
            return

        if kind is Kind.TASK_LIST:
            # If config also has a top-level "task", register it as the base task
            if "task" in cfg and isinstance(cfg["task"], str):
                base_name = cfg["task"]
                if base_name not in index:
                    index[base_name] = Entry(
                        name=base_name,
                        kind=Kind.TASK,
                        yaml_path=path,
                        tags=TaskIndex._str_to_set(cfg.get("tag")),
                        cfg=cfg,
                    )
                    TaskIndex._register_tags(base_name, cfg.get("tag"), index)

            # Register each task in task_list
            base_tag = cfg.get("tag")
            for entry in cfg["task_list"]:
                task_name = entry["task"] if isinstance(entry, dict) else entry
                if task_name in index:
                    log.warning(
                        f"Duplicate task name '{task_name}' found. "
                        f"Already registered from: {index[task_name].yaml_path}. "
                        f"Skipping duplicate from: {path}"
                    )
                    continue
                # Combine base tag with per-entry tag
                entry_tag = entry.get("tag") if isinstance(entry, dict) else None
                combined_tags = TaskIndex._str_to_set(base_tag) | TaskIndex._str_to_set(
                    entry_tag
                )
                index[task_name] = Entry(
                    name=task_name,
                    kind=Kind.TASK,
                    yaml_path=path,
                    tags=combined_tags,
                    cfg=cfg,
                )
                # Register both base config's tag and per-entry tag
                TaskIndex._register_tags(task_name, base_tag, index)
                TaskIndex._register_tags(task_name, entry_tag, index)
            return

    @staticmethod
    def _register_tags(
        task: str,
        tags: str | list[str] | None,
        index: dict[str, Entry],
    ) -> None:
        if not tags:
            return
        for tag in tags if isinstance(tags, list) else [tags]:
            entry = index.setdefault(
                tag,
                Entry(name=tag, kind=Kind.TAG, yaml_path=None, tags=set()),
            )
            entry.tags.add(task)

    @staticmethod
    def _kind_of(cfg: dict) -> Kind:
        if "class" in cfg:
            return Kind.PY_TASK
        if "group" in cfg:
            return Kind.GROUP
        if "task_list" in cfg:
            return Kind.TASK_LIST
        if "task" in cfg:
            return Kind.GROUP if isinstance(cfg["task"], list) else Kind.TASK
        msg = "Unknown config shape"
        raise ValueError(msg) from None

    @staticmethod
    def _str_to_set(tags: str | list[str] | None = None) -> set[str]:
        """Convert a string or list of strings to a set of strings."""
        return (
            set(tags)
            if isinstance(tags, list)
            else {tags}
            if isinstance(tags, str)
            else set()
        )
