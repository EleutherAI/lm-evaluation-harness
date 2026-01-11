from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from lm_eval.tasks._config_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

log = logging.getLogger(__name__)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


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
    # Hierarchical task support
    parent: str | None = (
        None  # parent path for inline children (e.g., "mmlu" for "mmlu::stem")
    )
    ref_target: str | None = None  # for children with ref: points to external entry
    tag_ref: str | None = None  # for children with tag: expands to tagged tasks


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
                    cfg = load_yaml(
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
        parent_path: str | None = None,
    ) -> None:
        kind = TaskIndex._kind_of(cfg)
        match kind:
            case Kind.GROUP:
                grp_name = cfg["group"]
                # Build full path for hierarchical addressing
                full_path = f"{parent_path}::{grp_name}" if parent_path else grp_name

                if full_path in index:
                    log.debug(
                        f"Duplicate group name '{full_path}' found. "
                        f"Already registered from: {index[full_path].yaml_path}. "
                        f"Skipping duplicate from: {path}"
                    )
                    return
                index[full_path] = Entry(
                    name=full_path,
                    kind=Kind.GROUP,
                    yaml_path=path,
                    tags=TaskIndex._str_to_set(cfg.get("tag")),
                    cfg=cfg,
                    parent=parent_path,
                )

                # Process inline children if present
                if "children" in cfg:
                    TaskIndex._process_children(cfg["children"], full_path, path, index)

            case Kind.TASK | Kind.PY_TASK:
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

            case Kind.TASK_LIST:
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
                    else:
                        log.warning(
                            f"Duplicate task name '{base_name}' found. "
                            f"Already registered from: {index[base_name].yaml_path}. "
                            f"Skipping duplicate from: {path}"
                        )

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
                    combined_tags = TaskIndex._str_to_set(base_tag, entry_tag)
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
        match cfg:
            # Python task: has 'class' key (check before 'task' since PY_TASK may have both)
            case {"class": _}:
                return Kind.PY_TASK
            # Group configs have task: list[str | dict]
            case {"group": _}:
                return Kind.GROUP
            case {"task_list": _}:
                return Kind.TASK_LIST
            case {"task": _}:
                return Kind.TASK
            case _:
                raise ValueError("Unknown config shape")

    @staticmethod
    def _str_to_set(*args) -> set[str]:
        """Convert a string or list of strings to a set of strings."""
        result = set()
        if not args:
            return result
        for t in args:
            match t:
                case str(): result.add(t),  # fmt: skip
                case list(): result.update(t)  # fmt: skip
        return result

    @staticmethod
    def _process_children(
        children: dict[str, Any],
        parent_path: str,
        yaml_path: Path,
        index: dict[str, Entry],
    ) -> None:
        """Process children definitions within a group.

        Children can be:
        - Task reference: {task: "task_name", ...overrides}
        - Subgroup: {group: "name", task: [...], children: {...}}

        Not supported (use task_list for task definitions):
        - Inline task definitions
        - Bare string values
        """
        for child_name, child_cfg in children.items():
            if not isinstance(child_cfg, dict):
                log.warning(
                    f"Invalid child config for '{child_name}' in '{parent_path}': "
                    f"expected dict with 'task:' or 'group:' key, got {type(child_cfg).__name__}"
                )
                continue

            child_path = f"{parent_path}::{child_name}"

            if child_path in index:
                log.debug(f"Duplicate child '{child_path}' found, skipping.")
                continue

            if "group" in child_cfg:
                # Subgroup definition - process recursively
                TaskIndex.process_cfg(
                    child_cfg, yaml_path, index, parent_path=parent_path
                )
            elif "task" in child_cfg and isinstance(child_cfg["task"], str):
                # Task reference (with optional overrides)
                index[child_path] = Entry(
                    name=child_path,
                    kind=Kind.TASK,
                    yaml_path=yaml_path,
                    parent=parent_path,
                    ref_target=child_cfg["task"],
                    cfg=child_cfg,
                    tags=TaskIndex._str_to_set(child_cfg.get("tag")),
                )
            else:
                log.warning(
                    f"Invalid child '{child_name}' in '{parent_path}': "
                    f"must have 'task: <string>' or 'group: <name>'"
                )
