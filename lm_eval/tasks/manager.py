from __future__ import annotations

from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lm_eval import utils
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

        self.include_path = include_path
        self.metadata = metadata

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

        self._all_tasks = sorted(self._index.keys())
        self._all_subtasks = sorted(
            chain.from_iterable(buckets[k] for k in {Kind.TASK, Kind.PY_TASK})
        )
        self._all_groups = sorted(buckets[Kind.GROUP])
        self._all_tags = sorted(buckets[Kind.TAG])

    # ---------------------------------------------------------------- properties
    @property
    def all_tasks(self) -> list[str]:
        return self._all_tasks

    @property
    def all_groups(self) -> list[str]:
        return self._all_groups

    @property
    def all_subtasks(self) -> list[str]:
        return self._all_subtasks

    @property
    def all_tags(self) -> list[str]:
        return self._all_tags

    @property
    def task_index(self) -> dict[str, Entry]:
        return self._index

    # ---------------------------------------------------------------- name checks
    def _name_is_registered(self, name: str) -> bool:
        return name in self._index

    def _name_is_task(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.TASK

    def _name_is_tag(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.TAG

    def _name_is_group(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.GROUP

    def _name_is_python_task(self, name: str) -> bool:
        return self._name_is_registered(name) and self._index[name].kind == Kind.PY_TASK

    # ---------------------------------------------------------------- utility
    def match_tasks(self, task_list: list[str]) -> list[str]:
        """Match task names using glob patterns."""
        return utils.pattern_match(task_list, self.all_tasks)

    def list_all_tasks(
        self,
        list_groups: bool = True,
        list_tags: bool = True,
        list_subtasks: bool = True,
    ) -> str:
        """Generate a markdown table listing all available tasks."""
        from pytablewriter import MarkdownTableWriter

        def sanitize_path(path):
            if path is None:
                return "---"
            path_str = str(path)
            if "lm_eval/tasks/" in path_str:
                return "lm_eval/tasks/" + path_str.split("lm_eval/tasks/")[-1]
            return path_str

        group_table = MarkdownTableWriter()
        group_table.headers = ["Group", "Config Location"]
        gt_values = []
        for g in self.all_groups:
            entry = self._index[g]
            path = sanitize_path(entry.yaml_path)
            gt_values.append([g, path])
        group_table.value_matrix = gt_values

        tag_table = MarkdownTableWriter()
        tag_table.headers = ["Tag"]
        tag_table.value_matrix = [[t] for t in self.all_tags]

        subtask_table = MarkdownTableWriter()
        subtask_table.headers = ["Task", "Config Location", "Output Type"]
        st_values = []
        for t in self.all_subtasks:
            entry = self._index[t]
            path = entry.yaml_path
            output_type = ""

            if path is not None:
                config = utils.load_yaml_config(str(path), mode="simple")
                if "output_type" in config:
                    output_type = config["output_type"]
                elif "include" in config:
                    include_path = str(path.parent / config["include"])
                    include_config = utils.load_yaml_config(include_path, mode="simple")
                    if "output_type" in include_config:
                        output_type = include_config["output_type"]

            path = sanitize_path(path)
            st_values.append([t, path, output_type])
        subtask_table.value_matrix = st_values

        result = "\n"
        if list_groups:
            result += group_table.dumps() + "\n\n"
        if list_tags:
            result += tag_table.dumps() + "\n\n"
        if list_subtasks:
            result += subtask_table.dumps() + "\n\n"
        return result

    # ---------------------------------------------------------------- core API
    def _entry(self, name: str) -> Entry:
        if name not in self._index:
            raise KeyError(f"Unknown task/group/tag: {name}")
        return self._index[name]

    def load_spec(self, spec: str | dict[str, Any]):
        """Spec can be:
        • str task / group / tag name (registered)
        • dict inline overrides {'task': 'hellaswag', 'num_fewshot': 5}
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

    def load_task_or_group(self, task_list: str | list[str]) -> dict:
        """Load tasks/groups and return a merged dictionary.

        :param task_list: Single task name or list of task names
        :return: Dictionary of task objects (possibly nested for groups)
        """
        import collections

        if isinstance(task_list, str):
            task_list = [task_list]

        # Each load_spec call returns a dict (possibly nested for groups)
        # We merge them using ChainMap (like the original implementation)
        return dict(collections.ChainMap(*[self.load_spec(s) for s in task_list]))

    def load_config(self, config: dict) -> dict:
        """Load a task from an inline config dict."""
        return self.load_spec(config)


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
