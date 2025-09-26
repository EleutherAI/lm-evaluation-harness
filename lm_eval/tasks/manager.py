from __future__ import annotations

from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lm_eval.api.task import Task
from lm_eval.tasks.factory import TaskFactory
from lm_eval.tasks.index import Entry, Kind, TaskIndex
from lm_eval.utils import setup_logging


if TYPE_CHECKING:
    from lm_eval.api.task import Task


class TaskManager:
    def __init__(
        self,
        verbosity: str | None = None,
        include_path: str | Path | list[str | Path] | None = None,
        include_defaults: bool = True,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if verbosity:
            setup_logging(verbosity)

        index = TaskIndex()
        self._factory = TaskFactory(meta=metadata)

        all_paths: list[Path] = []
        if include_defaults:
            all_paths.append(Path(__file__).parent)
        if include_path:
            all_paths += [
                Path(p)
                for p in (
                    include_path
                    if isinstance(include_path, (list, tuple))
                    else [include_path]
                )
            ]

        self._index = index.build(all_paths)

        buckets = defaultdict(list)
        for k, e in self._index.items():
            buckets[e.kind].append(k)

        self._all_tasks = sorted(
            chain.from_iterable(buckets[k] for k in {Kind.TASK, Kind.PY_TASK})
        )
        self._all_groups = sorted(buckets[Kind.GROUP])
        self._all_tags = sorted(buckets[Kind.TAG])

    def _entry(self, name: str) -> Entry:
        if name not in self._index:
            raise KeyError(f"Unknown task/group/tag: {name}")
        return self._index[name]

    def load_spec(self, spec: str | dict[str, Any]):
        """Spec can be:
        • str  task / group / tag name (registered)
        • dict inline overrides   {'task': 'hellaswag', 'num_fewshot': 5}
        """
        if isinstance(spec, str):
            entry = self._entry(spec)
            return self._factory.build(entry, overrides=None, registry=self._index)

        if isinstance(spec, dict):
            # inline dict => find base entry, then pass overrides
            name = spec["task"]
            entry = self._entry(name)
            return self._factory.build(entry, overrides=spec, registry=self._index)

        raise TypeError("spec must be str or dict")

    def load_task_or_group(self, task_list: str | list[str]):
        return (
            [self.load_spec(s) for s in task_list]
            if isinstance(task_list, list)
            else [self.load_spec(task_list)]
        )


def get_task_dict(
    task_name_list: str | list[str | dict | Task],
    task_manager: TaskManager | None = None,
):
    if not task_manager:
        task_manager = TaskManager()
    else:
        assert isinstance(task_manager, TaskManager)

    return {
        task_name: task_manager.load_spec(task_name)
        if isinstance(task_name, str)
        else task_name
        for task_name in task_name_list
    }
