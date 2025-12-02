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
    # Hierarchical task support
    parent: str | None = (
        None  # parent path for inline children (e.g., "mmlu" for "mmlu::stem")
    )
    ref_target: str | None = None  # for children with ref: points to external entry
    tag_ref: str | None = None  # for children with tag: expands to tagged tasks


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
        parent_path: str | None = None,
    ) -> None:
        kind = TaskIndex._kind_of(cfg)
        if kind is Kind.GROUP:
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

    @staticmethod
    def _process_children(
        children: dict[str, Any],
        parent_path: str,
        yaml_path: Path,
        index: dict[str, Entry],
    ) -> None:
        """Process inline children definitions within a group.

        Children can be:
        - Inline task: dict with task config fields (dataset_path, doc_to_text, etc.)
        - Inline subgroup: dict with 'children' key
        - External ref: dict with 'ref' key pointing to existing entry
        - Tag expansion: dict with 'tag' key to expand tagged tasks
        """
        for child_name, child_cfg in children.items():
            if not isinstance(child_cfg, dict):
                log.warning(
                    f"Invalid child config for '{child_name}' in '{parent_path}': "
                    f"expected dict, got {type(child_cfg).__name__}"
                )
                continue

            child_path = f"{parent_path}::{child_name}"

            if child_path in index:
                log.debug(f"Duplicate child '{child_path}' found, skipping.")
                continue

            if "ref" in child_cfg:
                # External reference - register with ref_target for build-time resolution
                index[child_path] = Entry(
                    name=child_path,
                    kind=Kind.GROUP,  # Assume group, will resolve at build time
                    yaml_path=yaml_path,
                    parent=parent_path,
                    ref_target=child_cfg["ref"],
                    cfg=child_cfg,
                    tags=TaskIndex._str_to_set(child_cfg.get("tag")),
                )

            elif "tag" in child_cfg:
                # Tag expansion - register with tag_ref for build-time expansion
                index[child_path] = Entry(
                    name=child_path,
                    kind=Kind.TAG,
                    yaml_path=yaml_path,
                    parent=parent_path,
                    tag_ref=child_cfg["tag"],
                    cfg=child_cfg,
                    tags=TaskIndex._str_to_set(child_cfg.get("tag")),
                )

            elif "children" in child_cfg:
                # Nested inline group - recurse
                nested_cfg = {**child_cfg, "group": child_name}
                TaskIndex.process_cfg(
                    nested_cfg, yaml_path, index, parent_path=parent_path
                )

            else:
                # Inline task definition
                task_cfg = {**child_cfg, "task": child_path}
                index[child_path] = Entry(
                    name=child_path,
                    kind=Kind.TASK,
                    yaml_path=yaml_path,
                    parent=parent_path,
                    cfg=task_cfg,
                    tags=TaskIndex._str_to_set(child_cfg.get("tag")),
                )
                # Register tags for inline tasks
                TaskIndex._register_tags(child_path, child_cfg.get("tag"), index)
