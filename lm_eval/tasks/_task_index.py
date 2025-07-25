# lm_eval/task_index.py  (continued)
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from lm_eval.tasks._config_loader import load_yaml as load_cfg


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class TaskKind(Enum):
    TASK = auto()  # YAML task, or task_list entry
    PY_TASK = auto()  # Python‑defined, via "class"
    GROUP = auto()
    TAG = auto()
    TASK_LIST = auto()


@dataclass
class TaskEntry:
    name: str
    kind: TaskKind
    yaml_path: Path | None  # None for generated / py‑only entries
    tags: set[str] = field(default_factory=set)
    task_list_path: Path | None = None  # only for GROUP / TAG when lazy‑loaded


log = logging.getLogger(__name__)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


class TaskIndexBuilder:
    """Walks one or more directories, parses YAML quickly (functions unresolved),
    and produces a mapping {task_name: TaskEntry}.
    """

    def __init__(self, *, metadata: dict | None = None) -> None:
        self._metadata = metadata or {}

    # ------------- public API --------------------------------------------------
    def build(
        self,
        paths: Iterable[Path],
        # include_defaults: bool = True,
    ) -> dict[str, TaskEntry]:
        index: dict[str, TaskEntry] = {}
        for root in paths:
            for yaml_path in self._iter_yaml_files(root):
                try:
                    cfg = load_cfg(
                        yaml_path,
                        resolve_functions=False,
                        resolve_includes=False,
                    )
                except Exception as err:
                    log.debug("Skip %s (%s)", yaml_path, err)
                    continue

                self._process_cfg(cfg, yaml_path, index)
        return index

    # ------------- helpers -----------------------------------------------------
    def _iter_yaml_files(self, root: Path):
        yield from (
            p
            for p in root.glob("**/*.yaml")
            if not any(part in _IGNORE_DIRS for part in p.parts)
        )

    # ---------------------------------------------------------------------------
    def _process_cfg(
        self,
        cfg: dict,
        path: Path,
        index: dict[str, TaskEntry],
    ) -> None:
        kind = self._kind_of(cfg)
        if kind is TaskKind.GROUP:
            grp_name = cfg["group"]
            index[grp_name] = TaskEntry(
                name=grp_name,
                kind=TaskKind.GROUP,
                yaml_path=path,
                tags=set(cfg.get("tag", [])),
            )
            return

        if kind is TaskKind.PY_TASK:
            name = cfg["task"]
            index[name] = TaskEntry(
                name=name,
                kind=TaskKind.PY_TASK,
                yaml_path=None,
                tags=set(cfg.get("tag", [])),
            )
            self._register_tags(name, cfg.get("tag", []), index)
            return

        if kind is TaskKind.TASK:
            name = cfg["task"]
            index[name] = TaskEntry(
                name=name,
                kind=TaskKind.TASK,
                yaml_path=path,
                tags=set(cfg.get("tag", [])),
            )
            self._register_tags(name, cfg.get("tag", []), index)
            return

        if kind is TaskKind.TASK_LIST:
            for entry in cfg["task_list"]:
                task_name = entry["task"] if isinstance(entry, dict) else entry
                index[task_name] = TaskEntry(
                    name=task_name,
                    kind=TaskKind.TASK,
                    yaml_path=path,
                    tags=set(entry.get("tag", []))
                    if isinstance(entry, dict)
                    else set(),
                )
                self._register_tags(task_name, entry.get("tag", []), index)
            return

    # ---------------------------------------------------------------------------
    def _register_tags(self, task: str, tags, index) -> None:
        for tag in tags if isinstance(tags, list) else [tags]:
            if not tag:
                continue
            entry = index.setdefault(
                tag,
                TaskEntry(name=tag, kind=TaskKind.TAG, yaml_path=None, tags=set()),
            )
            entry.tags.add(task)  # mutate ok; dataclass not frozen for TAG

    @staticmethod
    def _kind_of(cfg: dict) -> TaskKind:
        if "class" in cfg:
            return TaskKind.PY_TASK
        if "task_list" in cfg:
            return TaskKind.TASK_LIST
        if "task" in cfg:
            return TaskKind.GROUP if isinstance(cfg["task"], list) else TaskKind.TASK
        msg = "Unknown config shape"
        raise ValueError(msg)
