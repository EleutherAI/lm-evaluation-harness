from __future__ import annotations

import logging
import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import TypedDict, TypeVar, overload

from lm_eval.result_schema import _SampleCount


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lm_eval.api.group import Group
    from lm_eval.api.task import Task
    from lm_eval.result_schema import EvalResults, _TaskMetrics

_T = TypeVar("_T")

__all__ = ["run_task_tests", "torch_gather_object"]


def _log_rank_zero(logger: logging.Logger):
    class _RankZeroFilter(logging.Filter):
        """Suppress logs on non-zero ranks unless marked with ``all_ranks=True``.

        By default, all evaluator logs are rank-0-only. To let a specific log
        through on every rank, pass ``extra={"all_ranks": True}``.
        """

        def filter(self, record: logging.LogRecord) -> bool:
            if getattr(record, "all_ranks", False):
                return True
            return int(os.environ.get("RANK", "0")) == 0

    logger.addFilter(_RankZeroFilter())
    return logger


eval_logger = _log_rank_zero(logging.getLogger(__name__))


class _ResultAcc(TypedDict):
    """Accumulator for results of a single task."""

    task: Task
    logged_samples: list[Any]


@dataclass
class _EvalAcc:
    """Container for evaluation results."""

    # Core results: {task_name: {metric_key: value}}
    metrics: dict[str, _TaskMetrics[float]] = field(default_factory=dict)

    # Per-task metadata
    configs: dict[str, dict[str, str]] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    num_fewshot: dict[str, int] = field(default_factory=dict)
    higher_is_better: dict[str, dict[str, bool]] = field(default_factory=dict)

    # Sample-level data (optional)
    samples: dict[str, list[Any]] = field(default_factory=dict)

    # Original vs effective sample counts per task
    n_samples: dict[str, _SampleCount] = field(default_factory=dict)

    # All groups (for aggregation and formatting)
    groups: Mapping[str, Group] = field(default_factory=dict)

    def collect(self) -> tuple[dict[str, _TaskMetrics], dict[str, _TaskMetrics]]:
        """Collect metrics into task_results and group_results.

        All entries go into task_results. Groups with aggregation also go
        into group_results.
        """
        task_results = {}
        group_results = {}
        for name, metrics in self.metrics.items():
            task_results[name] = dict(metrics)
            if name in self.groups and self.groups[name].has_aggregation:
                group_results[name] = dict(metrics)
        return task_results, group_results

    def _to_eval_results(
        self, *, samples: dict[str, list] | None = None
    ) -> EvalResults:
        """Assemble the final EvalResults dict from accumulated data.

        Args:
            samples: Optional pre-processed samples dict.

        Returns:
            EvalResults: TypedDict ready for serialisation / display.
        """
        task_data, group_data = self.collect()

        all_groups: list[Group] = list(self.groups.values())
        subtask_list = {group.name: group.child_names for group in all_groups}

        higher_is_better = dict(self.higher_is_better)
        _propagate_higher_is_better_(all_groups, higher_is_better)

        num_fewshot = dict(self.num_fewshot)
        _propagate_num_fewshot_(all_groups, num_fewshot)

        results_dict: EvalResults = {
            "results": task_data,
            **({"groups": group_data} if group_data else {}),
            "group_subtasks": subtask_list,
            "configs": dict(sorted(self.configs.items())),
            "versions": dict(sorted(self.versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": dict(self.n_samples),
        }
        if samples is not None:
            results_dict["samples"] = dict(samples)

        return results_dict


def _print_writeout(task: Task) -> None:
    for inst in task.instances:
        # print the prompt for the first few documents
        if inst.doc_id is not None and inst.doc_id < 1:
            eval_logger.info(
                "Task: %s; document %s; context prompt (starting on next line):\
    \n%s\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n%s\n(end of target on previous line)",
                task,
                inst.doc_id,
                inst.args[0],
                task.doc_to_target(inst.doc),
            )
            eval_logger.info("Request: %s", inst)
            break


def _get_sample_size(task, limit: float | None) -> int | None:
    if limit is None:
        return None
    return math.ceil(len(task.eval_docs) * limit) if limit < 1.0 else int(limit)


def _find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    """Search upward in the directory tree to a maximum of three layers to find and return the package root (containing the 'tests' folder)."""
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(
        f"Unable to find package root within {max_layers} upwards" + f"of {start_path}"
    )


def run_task_tests(task_list: list[str]):
    """Find the package root and run the tests for the given tasks."""
    import pytest

    package_root = _find_test_root(start_path=pathlib.Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(
            f"Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}"
        )


def _compute_task_aggregations(
    task: Task,
    bootstrap_iters: int | None = 100000,
) -> tuple[dict[str, Any], int]:
    """Compute aggregated metrics from scorer-internal reduced results.

    Delegates to ``task.aggregate()`` which calls each scorer's
    ``aggregate()`` method.
    """
    return task.aggregate(bootstrap_iters)


def _agg_and_collect(
    eval_results_acc: Mapping[str, _ResultAcc],
    groups: Mapping[str, Group] | None = None,
    bootstrap_iters: int | None = 100000,
) -> _EvalAcc:
    """Collect and aggregate task results into _EvalAcc container.

    Args:
        eval_results_acc: Accumulated metrics from evaluation.
            Format: {task_name: {"task": Task, "logged_samples": [...]}}
        groups: Dict of group name -> Group objects
        bootstrap_iters: Number of bootstrap iterations for stderr calculation

    Returns:
        _EvalAcc with all results consolidated
    """
    result = _EvalAcc()
    result.groups = groups or {}

    for task_name, acc in eval_results_acc.items():
        task = acc["task"]

        agg_metrics, sample_len = _compute_task_aggregations(task, bootstrap_iters)
        task_config = dict(task.dump_config())

        result.metrics[task_name] = {
            "name": task_name,
            "alias": task_config.get("task_alias") or task.task_name,
            "sample_len": sample_len,
            **agg_metrics,
        }
        result.configs[task_name] = task_config
        result.versions[task_name] = task.VERSION
        result.num_fewshot[task_name] = task_config.get("num_fewshot", 0)
        result.higher_is_better[task_name] = task.higher_is_better()
        result.samples[task_name] = acc["logged_samples"]

        # Compute n_samples: effective comes from sample_len (actual evaluated count)
        original = len(task.eval_docs)
        result.n_samples[task_name] = _SampleCount(
            original=original, effective=sample_len
        )

    return result


def _aggregate_groups(
    results: _EvalAcc,
) -> _EvalAcc:
    """Compute aggregated metrics for groups.

    Processes groups bottom-up (children before parents) and delegates
    aggregation to each Group's aggregate() method.

    Args:
        results: _EvalAcc from collect_results

    Returns:
        Same _EvalAcc with group metrics added
    """
    # Collect all groups in bottom-up order (children before parents)
    all_groups = _collect_groups_bottom_up(results.groups)

    for group in all_groups:
        # Each group aggregates its own metrics from leaf tasks
        results.metrics[group.name] = group.aggregate(results.metrics)
        # Set group version from metadata if available
        results.versions[group.name] = group.version

    return results


def _get_root_groups(groups: Mapping[str, Group]) -> list[Group]:
    """Find groups that aren't children of any other group.

    We assume all groups have unique names and no cycles exist.
    These are the "top-level" groups for traversal.
    """
    # Collect all group names that appear as children of other groups
    child_names: set[str] = set()
    for group in groups.values():
        for subgroup in group.get_all_groups(recursive=False):
            child_names.add(subgroup.name)

    # Root groups are those not in child_names
    return [g for name, g in groups.items() if name not in child_names]


def _collect_groups_bottom_up(groups: Mapping[str, Group]) -> list[Group]:
    """Collect all groups in bottom-up order (children before parents).

    This is a post-order traversal that ensures subgroups are processed
    before their parent groups.
    """
    from lm_eval.api.group import Group

    result: list[Group] = []
    visited: set[str] = set()

    def visit(group: Group) -> None:
        if group.name in visited:
            return
        # Visit children first (post-order)
        for child in group:
            if isinstance(child, Group):
                visit(child)
        # Then add this group
        visited.add(group.name)
        result.append(group)

    # Start from root groups and traverse down
    for root in _get_root_groups(groups):
        visit(root)

    return result


def _process_results(
    eval_results_acc: Mapping[str, _ResultAcc],
    groups: Mapping[str, Group] | None = None,
    bootstrap_iters: int | None = 100000,
) -> _EvalAcc:
    """Process evaluation results.

    Args:
        eval_results_acc: Accumulated metrics from evaluation.
            Format: {task_name: {"task": Task, "logged_samples": [...]}}
            Task objects must have scorer.reduced_docs populated
            (via task.process_instances() or task.import_reduced()).
        groups: Dict of group name -> Group
        bootstrap_iters: Number of bootstrap iterations for stderr calculation

    Returns:
        _EvalAcc dataclass with:
            - metrics: Dict of task/group metrics
            - configs: Task configurations
            - versions: Task versions
            - num_fewshot: Number of few-shot examples
            - higher_is_better: Metric direction info
            - samples: Sample-level results
            - n_samples: Original and effective sample counts per task
            - groups: Groups dict for traversal

    Example:
        ```python
        loaded = task_manager.load(["arc", "hellaswag"])

        # Run evaluation (populates scorer.reduced_docs)
        eval_results_acc = {
            name: {"task": t, "logged_samples": []}
            for name, t in loaded["tasks"].items()
        }

        results = _process_results(eval_results_acc, loaded["groups"])

        # Convert to EvalResults dict
        eval_results = results._to_eval_results()
        ```
    """
    # Collect task results (includes aggregation)
    results = _agg_and_collect(eval_results_acc, groups or {}, bootstrap_iters)

    # Aggregate group metrics
    return _aggregate_groups(results)


def _propagate_num_fewshot_(
    all_groups: Iterable[Group], num_fewshot: dict[str, int]
) -> None:
    for group in all_groups:
        values = {num_fewshot[c] for c in group.child_names if c in num_fewshot}
        if len(values) == 1:
            num_fewshot[group.name] = values.pop()


def _propagate_higher_is_better_(
    all_groups: Iterable[Group], higher_is_better: dict[str, dict[str, bool]]
) -> None:
    for group in all_groups:
        _higher_is_better = {}
        for child in group.child_names:
            if child in higher_is_better:
                for m, h in higher_is_better[child].items():
                    if m not in _higher_is_better:
                        _higher_is_better[m] = h
                    elif _higher_is_better[m] is not None and _higher_is_better[m] != h:
                        eval_logger.warning(
                            "Higher_is_better values for metric %s in group %s are not consistent. Defaulting to None.",
                            m,
                            group.name,
                        )
                        _higher_is_better[m] = None
        if _higher_is_better:
            higher_is_better[group.name] = _higher_is_better


def _log_selected_tasks_(
    task_dict: dict,
    groups: dict[str, Group],
) -> None:
    """Log selected tasks with hierarchy information."""

    def get_task_path(task_name: str) -> str:
        """Get display path for a task from its config metadata."""
        task = task_dict.get(task_name)
        if task is None:
            return "N/A"
        source = task.config.metadata.get("config_source")
        if not source or source == "inline":
            return "N/A"
        return source

    eval_logger.info("Selected tasks:")

    # Find root groups (not children of other groups)
    all_children = set()
    for group in groups.values():
        all_children.update(group.child_names)
    root_groups = [name for name in groups if name not in all_children]

    # Log groups hierarchically
    logged_tasks = set()

    def log_group(group_name: str, indent: int = 0):
        if group_name not in groups:
            return
        group = groups[group_name]
        pad = "  " * indent
        eval_logger.info("%sGroup: %s", pad, group_name)

        for child in group.child_names:
            if child in groups:
                log_group(child, indent + 1)
            elif child in task_dict:
                child_pad = "  " * (indent + 1)
                path = get_task_path(child)
                eval_logger.info("%sTask: %s (%s)", child_pad, child, path)
                logged_tasks.add(child)

    for root in sorted(root_groups):
        log_group(root)

    # Log standalone tasks (not in any group)
    for task_name in sorted(task_dict.keys()):
        if task_name not in logged_tasks:
            path = get_task_path(task_name)
            eval_logger.info("Task: %s (%s)", task_name, path)


@overload
def torch_gather_object(
    obj: _T, rank: int, world_size: int, dst: int = 0
) -> list[_T]: ...
@overload
def torch_gather_object(
    obj: _T, rank: int, world_size: int, dst: int = ...
) -> None: ...
def torch_gather_object(
    obj: _T, *, rank: int, world_size: int, dst: int = 0
) -> list[_T] | None:
    """Gather a Python object from all ranks to ``dst``.

    Returns a list of objects (one per rank) on ``dst``, ``None`` on all other
    ranks.  When ``world_size == 1`` this is a no-op that returns ``[obj]``.
    """
    if world_size <= 1:
        return [obj]
    import torch.distributed as dist

    result = [None] * world_size if rank == dst else None
    dist.gather_object(obj=obj, object_gather_list=result, dst=dst)  # type:ignore[possibly-missing-attribute]
    return cast("list[_T] | None", result)


def _merge_rank_metrics(
    all_rank_data: list[dict], task_name: str
) -> dict[str, dict[int, dict[str, float]]]:
    """Merge per-task reduced docs from all ranks.

    Each rank exports ``{scorer: {doc_id: {metric: value}}}``.
    Merging is a dict update — order-independent since doc IDs are unique
    per rank.
    """
    merged: dict[str, dict[int, dict[str, float]]] = {}
    for rank_data in all_rank_data:
        if task_name not in rank_data:
            continue
        for scorer_name, docs in rank_data[task_name].items():
            merged.setdefault(scorer_name, {}).update(docs)
    return merged


def _build_logged_samples(
    task: Task,
    samples: dict[str, list[int]] | None,
    task_name: str,
) -> list[dict[str, Any]]:
    """Build per-document sample logs for a task.

    Reads fields directly from Instance objects, reduced metrics from
    ``scorer.reduced_docs``, and per-repeat raw scores from
    ``scorer.raw_docs``.  Instances are already filtered by
    rank/limit/world_size during ``build_all_requests``.
    """
    import json

    from lm_eval.api.utils import group_by_doc_id
    from lm_eval.utils import handle_non_serializable, hash_string

    logged: list[dict[str, Any]] = []

    instances_by_doc_id = group_by_doc_id(task.instances)

    indices = samples.get(task_name, None) if samples is not None else None

    for scorer in task.scorers or []:
        for doc_id, reqs in instances_by_doc_id.items():
            first = reqs[0]
            doc_id_true = indices[doc_id] if indices else doc_id
            target = first.target
            rd = scorer.reduced_docs.get(doc_id)

            per_doc_metrics = rd or {}

            example = {
                "doc_id": doc_id_true,
                "doc": first.doc,
                "target": target,
                "arguments": [req.args for req in reqs],
                "resps": [req.resps for req in reqs],
                "filtered_resps": [
                    req.filtered_resps.get(scorer.name, req.resps[0]) for req in reqs
                ],
                "filter": scorer.name,
                "metrics": list(per_doc_metrics.keys()),
                "doc_hash": hash_string(
                    json.dumps(
                        first.doc,
                        indent=2,
                        default=handle_non_serializable,
                        ensure_ascii=False,
                    )
                ),
                "prompt_hash": hash_string(first.arguments[0]),
                "target_hash": hash_string(str(target)),
                **per_doc_metrics,
            }
            # Include per-repeat scores when repeats > 1
            raw = scorer.raw_docs.get(doc_id)
            if raw:
                repeats = {k: v for k, v in raw.scores.items() if len(v) > 1}
                if repeats:
                    example["scores_per_repeat"] = repeats
            logged.append(example)

    return logged


def _handle_back_comp(
    nested_dict: dict,
) -> tuple[dict[str, Group], dict[str, Task]]:
    """Handle backward compatibility for the legacy nested-dict task format.

    The legacy ``load_task_or_group`` returns
    ``{ConfigurableGroup: {task_name: Task, ...}, task_name: Task, ...}``.

    This converts it into the ``(groups, tasks)`` tuple expected by the
    new evaluator code path.
    """
    from lm_eval.api.group import Group

    groups: dict[str, Group] = {}
    tasks: dict[str, Task] = {}

    for key, value in nested_dict.items():
        if isinstance(key, Group):
            # key is a ConfigurableGroup/Group, value is {task_name: Task}
            groups[key.name] = key
            if isinstance(value, dict):
                tasks.update(value)
        elif isinstance(key, str):
            # Standalone task
            if isinstance(value, dict):
                # Nested dict of tasks (shouldn't normally happen, but be safe)
                tasks.update(value)
            else:
                tasks[key] = value

    return groups, tasks


def _has_bypass_metric(eval_tasks: dict[str, Task]) -> bool:
    """Check if any task uses the 'bypass' metric (requires log_samples=True)."""
    return any(
        m.name == "bypass"
        for task_obj in eval_tasks.values()
        for scorer in getattr(task_obj, "scorers", [])
        for m in (scorer.metrics or [])
    )
