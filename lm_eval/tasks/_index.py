from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from lm_eval.tasks._yaml_loader import load_yaml


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

log = logging.getLogger(__name__)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


class Kind(Enum):
    TASK = auto()
    PY_TASK = auto()  # Python-defined, via "class"
    GROUP = auto()
    TAG = auto()


@dataclass
class Entry:
    name: str
    kind: Kind
    yaml_path: Path | None  # None for generated / py-only entries
    cfg: dict[str, str] | None = None
    tags: set[str] = field(default_factory=set)


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
        match kind:
            case Kind.GROUP:
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
            case {"task": _}:
                return Kind.TASK
            case _:
                raise ValueError("Unknown config shape")

    @staticmethod
    def _str_to_set(*args) -> set[str]:
        """Convert a string or list of strings to a set of strings."""
        result = set()
        for t in args:
            if t is None:
                continue
            result.update([t] if isinstance(t, str) else t)
        return result
