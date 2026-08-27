from __future__ import annotations

import logging
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

from lm_eval.api.metrics import (
    mean,
    stderr_for_metric,
)
from lm_eval.result_schema import EvalResults, _SampleCount, _TaskMetrics
from lm_eval.utils import positional_deprecated


if TYPE_CHECKING:
    from lm_eval.api.group import Group
    from lm_eval.api.task import Task
    from lm_eval.tasks import TaskManager


eval_logger = logging.getLogger(__name__)


class ResultAcc(TypedDict):
    """Accumulator for results of a single task."""

    task: Task
    raw_metrics: dict[tuple[str, str], list[Any]]
    logged_samples: list[Any]


def print_writeout(task: Task) -> None:
    for inst in task.instances:
        # print the prompt for the first few documents
        if inst.doc_id is not None and inst.doc_id < 1:
            eval_logger.info(
                f"Task: {task}; document {inst.doc_id}; context prompt (starting on next line):\
    \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
            )
            eval_logger.info(f"Request: {str(inst)}")
            break


def get_sample_size(task, limit: int | float | None) -> int | None:
    if limit is not None:
        limit = (
            int(math.ceil(len(task.eval_docs) * limit)) if limit < 1.0 else int(limit)
        )
    return limit


@positional_deprecated
def find_test_root(start_path: pathlib.Path) -> pathlib.Path:
    """
    Search upward in the directory tree to a maximum of three layers
    to find and return the package root (containing the 'tests' folder)
    """
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


@positional_deprecated
def run_task_tests(task_list: list[str]):
    """
    Find the package root and run the tests for the given tasks
    """
    import pytest

    package_root = find_test_root(start_path=pathlib.Path(__file__))
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


@dataclass
class EvalAcc:
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
    groups: dict[str, Group] = field(default_factory=dict)

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
            samples: Optional pre-processed samples dict (caller handles
                multimodal hashing before passing in).

        Returns:
            EvalResults TypedDict ready for serialisation / display.
        """
        task_data, group_data = self.collect()

        all_groups: list[Group] = list(self.groups.values())
        subtask_list = {group.name: group.child_names for group in all_groups}

        higher_is_better = dict(self.higher_is_better)
        _propagate_higher_is_better(all_groups, higher_is_better)

        num_fewshot = dict(self.num_fewshot)
        _propagate_num_fewshot(all_groups, num_fewshot)

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


def _compute_task_aggregations(
    task: Task,
    raw_metrics: dict[tuple[str, str], list],
    bootstrap_iters: int | None = 100000,
) -> tuple[dict[str, Any], int]:
    """
    Compute aggregated metrics from raw per-sample metrics.

    Args:
        task: Task object (for aggregation functions)
        raw_metrics: {(metric_name, filter_key): [values]}
        bootstrap_iters: Number of bootstrap iterations for stderr

    Returns:
        (agg_metrics dict, sample_count)
    """
    agg_metrics: dict[str, Any] = {}
    sample_len = 0

    for (metric, filter_key), items in raw_metrics.items():
        try:
            agg_fn = task.aggregation()[metric]  # type: ignore[index]
        except KeyError:
            # Arbitrary metric without a defined aggregation function
            eval_logger.warning(
                f"[{task.task_name}] No aggregation function defined for metric {metric}. Using mean."
            )
            agg_fn = mean

        metric_key = f"{metric},{filter_key}"
        agg_metrics[metric_key] = agg_fn(items)
        sample_len = len(items)  # TODO: reflects only the last metric's count

        if isinstance(bootstrap_iters, int) and bootstrap_iters > 0:
            stderr_fn = stderr_for_metric(
                metric=agg_fn,
                bootstrap_iters=min(bootstrap_iters, 100)
                if metric in ["bleu", "chrf", "ter"]
                else bootstrap_iters,
            )
            agg_metrics[f"{metric}_stderr,{filter_key}"] = (
                stderr_fn(items) if (stderr_fn and len(items) > 1) else "N/A"
            )
        else:
            agg_metrics[f"{metric}_stderr,{filter_key}"] = "N/A"

    return agg_metrics, sample_len


def _collect_results(
    eval_results_acc: dict[str, ResultAcc],
    groups: dict[str, Group] | None = None,
    bootstrap_iters: int | None = 100000,
) -> EvalAcc:
    """
    Collect and aggregate task results into EvalAcc container.

    Args:
        eval_results_acc: Accumulated metrics from evaluation.
            Format: {task_name: {"task": Task, "raw_metrics": defaultdict, "logged_samples": []}}
        groups: Dict of group name -> Group objects
        bootstrap_iters: Number of bootstrap iterations for stderr calculation

    Returns:
        EvalAcc with all results consolidated
    """
    result = EvalAcc()
    result.groups = groups or {}

    for task_name, acc in eval_results_acc.items():
        task = acc["task"]

        # Compute aggregated metrics
        # TODO: note: currently assume all metrics are scalar-valued
        agg_metrics, sample_len = _compute_task_aggregations(
            task, acc["raw_metrics"], bootstrap_iters
        )

        # Get task config
        task_config = dict(task.dump_config())

        result.metrics[task_name] = {
            "name": task_name,
            "alias": task_config.get("task_alias", task_name),
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


def aggregate_groups(
    results: EvalAcc,
) -> EvalAcc:
    """
    Compute aggregated metrics for groups.

    Processes groups bottom-up (children before parents) and delegates
    aggregation to each Group's aggregate() method.

    Args:
        results: EvalAcc from collect_results

    Returns:
        Same EvalAcc with group metrics added
    """
    # Collect all groups in bottom-up order (children before parents)
    all_groups = _collect_groups_bottom_up(results.groups)

    for group in all_groups:
        # Each group aggregates its own metrics from leaf tasks
        results.metrics[group.name] = group.aggregate(results.metrics)
        # Set group version from metadata if available
        results.versions[group.name] = group.version

    return results


def _get_root_groups(groups: dict[str, Group]) -> list[Group]:
    """
    Find groups that aren't children of any other group.
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


def _collect_groups_bottom_up(groups: dict[str, Group]) -> list[Group]:
    """
    Collect all groups in bottom-up order (children before parents).

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
    eval_results_acc: dict[str, ResultAcc],
    groups: dict[str, Group] | None = None,
    bootstrap_iters: int | None = 100000,
) -> EvalAcc:
    """
    Process evaluation results.

    Args:
        eval_results_acc: Accumulated metrics from evaluation.
            Format: {task_name: {"task": Task, "raw_metrics": defaultdict, "logged_samples": []}}
        groups: Dict of group name -> Group
        bootstrap_iters: Number of bootstrap iterations for stderr calculation

    Returns:
        EvalAcc dataclass with:
        - metrics: Dict of task/group metrics
        - configs: Task configurations
        - versions: Task versions
        - num_fewshot: Number of few-shot examples
        - higher_is_better: Metric direction info
        - samples: Sample-level results
        - n_samples: Original and effective sample counts per task
        - groups: Groups dict for traversal

    Example usage:
        loaded = task_manager.load(['arc', 'hellaswag'])

        # Run evaluation (populates raw_metrics and logged_samples)
        eval_results_acc = {name: {"task": t, "raw_metrics": defaultdict(list), "logged_samples": []}
                           for name, t in loaded['tasks'].items()}

        results = _process_results(eval_results_acc, loaded['groups'])

        # Convert to EvalResults dict
        eval_results = results._to_eval_results()
    """
    # Collect task results (includes aggregation)
    results = _collect_results(eval_results_acc, groups or {}, bootstrap_iters)

    # Aggregate group metrics
    results = aggregate_groups(results)

    return results


def _propagate_num_fewshot(
    all_groups: list[Group], num_fewshot: dict[str, int]
) -> None:
    for group in all_groups:
        values = {num_fewshot[c] for c in group.child_names if c in num_fewshot}
        if len(values) == 1:
            num_fewshot[group.name] = values.pop()


def _propagate_higher_is_better(
    all_groups: list[Group], higher_is_better: dict[str, dict[str, bool]]
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
                            f"Higher_is_better values for metric {m} in group {group.name} are not consistent. Defaulting to None."
                        )
                        _higher_is_better[m] = None
        if _higher_is_better:
            higher_is_better[group.name] = _higher_is_better


def _log_selected_tasks(
    task_dict: dict,
    groups: dict[str, Group],
    task_manager: TaskManager,
) -> None:
    """Log selected tasks with hierarchy information."""
    from pathlib import Path

    # TODO: Add config info directly in Task object
    def get_task_path(task_name: str) -> str:
        """Get display path for a task."""
        if task_name not in task_manager.task_index:
            return "N/A"
        entry = task_manager.task_index[task_name]
        if not entry.yaml_path:
            return "N/A"
        yaml_path = Path(entry.yaml_path)
        tasks_dir = Path(__file__).parent / "tasks"
        try:
            return str(yaml_path.relative_to(tasks_dir))
        except ValueError:
            return str(yaml_path)

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
        eval_logger.info(f"{pad}Group: {group_name}")

        for child in group.child_names:
            if child in groups:
                log_group(child, indent + 1)
            elif child in task_dict:
                child_pad = "  " * (indent + 1)
                path = get_task_path(child)
                eval_logger.info(f"{child_pad}Task: {child} ({path})")
                logged_tasks.add(child)

    for root in sorted(root_groups):
        log_group(root)

    # Log standalone tasks (not in any group)
    for task_name in sorted(task_dict.keys()):
        if task_name not in logged_tasks:
            path = get_task_path(task_name)
            eval_logger.info(f"Task: {task_name} ({path})")


def _handle_back_comp(
    nested_dict: dict,
) -> tuple[dict[str, Group], dict[str, Task]]:
    """Handle backward compatibility for the legacy nested-dict task format.

    The legacy ``load_task_or_group`` returns::

        {ConfigurableGroup: {task_name: Task, ...}, task_name: Task, ...}

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
