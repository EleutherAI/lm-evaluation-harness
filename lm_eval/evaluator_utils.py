import logging
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict

from lm_eval.api.group import ConfigurableGroup, Group
from lm_eval.api.metrics import (
    mean,
    stderr_for_metric,
)
from lm_eval.api.task import Task
from lm_eval.utils import positional_deprecated


eval_logger = logging.getLogger(__name__)


class ResultAcc(TypedDict):
    """Accumulator for results of a single task."""

    task: Task
    raw_metrics: dict[tuple[str, str], list]
    logged_samples: list[Any]


def get_subtask_list(task_dict, task_root=None, depth=0):
    subtask_list = {}
    for group_obj, task_obj in task_dict.items():
        if isinstance(group_obj, ConfigurableGroup):
            # group_name = group_obj.group_name
            group_name = group_obj.group_name
        else:
            group_name = group_obj
        if isinstance(task_obj, dict):
            _subtask_list = get_subtask_list(
                task_obj, task_root=group_name, depth=depth + 1
            )
            if task_root:
                subtask_list.setdefault((task_root, depth), []).extend(
                    [
                        _task
                        for (_task, _depth) in _subtask_list.keys()
                        if (_depth - 1) == depth
                    ]
                )

            subtask_list = {**subtask_list, **_subtask_list}
        else:
            if isinstance(task_obj, ConfigurableGroup):
                # group_or_task_name = task_obj.group_name
                group_or_task_name = task_obj.group_name
            elif isinstance(task_obj, Task):
                # group_or_task_name = task_obj.task_name
                group_or_task_name = task_obj.task_name

            if task_root is None:
                subtask_list.setdefault((group_or_task_name, depth), [])
            else:
                subtask_list.setdefault((task_root, depth), []).append(
                    group_or_task_name
                )

    if depth == 0:
        _subtask_list = {}
        for group_key, task_list in subtask_list.items():
            group_name, depth = group_key
            _subtask_list[group_name] = task_list
        subtask_list = _subtask_list

    return subtask_list


def print_writeout(task) -> None:
    for inst in task.instances:
        # print the prompt for the first few documents
        if inst.doc_id < 1:
            eval_logger.info(
                f"Task: {task}; document {inst.doc_id}; context prompt (starting on next line):\
    \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
            )
            eval_logger.info(f"Request: {str(inst)}")


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
class EvalResults:
    """Container for evaluation results."""

    # Core results: {task_name: {metric_key: value}}
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-task metadata
    configs: dict[str, dict] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    num_fewshot: dict[str, int] = field(default_factory=dict)
    higher_is_better: dict[str, dict[str, bool]] = field(default_factory=dict)

    # Sample-level data (optional)
    samples: dict[str, list] = field(default_factory=dict)

    # All groups (for aggregation and formatting)
    groups: dict[str, "Group"] = field(default_factory=dict)


def compute_task_aggregations(
    task: "Task",
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
            eval_logger.info(
                f"[{task.task_name}] No aggregation function defined for metric {metric}. Using mean."
            )
            agg_fn = mean

        metric_key = f"{metric},{filter_key}"
        agg_metrics[metric_key] = agg_fn(items)
        sample_len = len(items)

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


def collect_results(
    eval_results_acc: dict[str, ResultAcc],
    groups: dict[str, "Group"] | None = None,
    bootstrap_iters: int | None = 100000,
) -> EvalResults:
    """
    Collect and aggregate task results into EvalResults container.

    Args:
        eval_results_acc: Accumulated metrics from evaluation.
            Format: {task_name: {"task": Task, "raw_metrics": defaultdict, "logged_samples": []}}
        groups: Dict of group name -> Group objects
        bootstrap_iters: Number of bootstrap iterations for stderr calculation

    Returns:
        EvalResults with all results consolidated
    """
    result = EvalResults()
    result.groups = groups or {}

    for task_name, acc in eval_results_acc.items():
        task = acc["task"]

        # Compute aggregated metrics
        agg_metrics, sample_len = compute_task_aggregations(
            task, acc["raw_metrics"], bootstrap_iters
        )

        # Get task config
        task_config = dict(task.dump_config())

        result.metrics[task_name] = {
            "alias": task_config.get("task_alias", task_name),
            "samples": sample_len,
            **agg_metrics,
        }
        result.configs[task_name] = task_config
        result.versions[task_name] = task.VERSION
        result.num_fewshot[task_name] = task_config.get("num_fewshot", 0)
        result.higher_is_better[task_name] = task.higher_is_better()
        result.samples[task_name] = acc["logged_samples"]

    return result


def aggregate_groups(
    results: EvalResults,
) -> EvalResults:
    """
    Compute aggregated metrics for groups.

    Processes groups bottom-up (children before parents) and delegates
    aggregation to each Group's aggregate() method.

    Args:
        results: EvalResults from collect_results_v2

    Returns:
        Same EvalResults with group metrics added
    """
    # Collect all groups in bottom-up order (children before parents)
    all_groups = _collect_groups_bottom_up(results.groups)

    for group in all_groups:
        # Each group aggregates its own metrics from leaf tasks
        results.metrics[group.name] = group.aggregate(results.metrics)
        # Set group version from metadata if available
        results.versions[group.name] = group.version

    return results


def get_root_groups(groups: dict[str, "Group"]) -> list["Group"]:
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


def _collect_groups_bottom_up(groups: dict[str, "Group"]) -> list["Group"]:
    """
    Collect all groups in bottom-up order (children before parents).

    This is a post-order traversal that ensures subgroups are processed
    before their parent groups.
    """
    result: list[Group] = []
    visited: set[str] = set()

    def visit(group: "Group") -> None:
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
    for root in get_root_groups(groups):
        visit(root)

    return result


def get_results_data(
    results: EvalResults,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Extract raw task and group results from EvalResults.
    Strips 'samples' count from each entry. No indentation or display formatting.

    Returns:
        (task_results, group_results) — raw metric dicts
    """
    task_results = {}
    group_results = {}

    for name, metrics in results.metrics.items():
        entry = dict(metrics)
        entry.pop("samples", None)
        task_results[name] = entry

        if name in results.groups:
            group = results.groups[name]
            if group.has_aggregation:
                group_results[name] = dict(entry)

    return task_results, group_results


def format_results(
    results: EvalResults,
    show_groups: bool = True,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """
    Format results for display with indentation.

    Args:
        results: EvalResults with metrics
        show_groups: Whether to include group aggregations in output

    Returns:
        Tuple of (task_results, group_results) formatted for display
    """
    task_results = {}
    group_results = {}

    def format_entry(name: str, metrics: dict, depth: int = 0) -> dict:
        indent = " " * depth + "- " if depth > 0 else ""
        formatted = dict(metrics)
        alias = formatted.get("alias", formatted.get("task_alias", name))
        formatted["alias"] = indent + alias
        formatted.pop("samples", None)
        return formatted

    def process_item(item: "Task | Group", depth: int) -> None:
        if isinstance(item, Group):
            name = item.name
            if name in results.metrics:
                formatted = format_entry(name, results.metrics[name], depth)
                task_results[name] = formatted

                if item.has_aggregation and show_groups:
                    group_results[name] = formatted.copy()

            # Process children
            for child in item:
                process_item(child, depth + 1)
        else:
            # Task
            name = item.task_name
            if name in results.metrics:
                task_results[name] = format_entry(name, results.metrics[name], depth)

    # Process root groups first (for hierarchy) - sorted for determinism
    for root in sorted(get_root_groups(results.groups), key=lambda g: g.name):
        process_item(root, 0)

    # Add all tasks not yet processed (standalone or from groups) - sorted for determinism
    for task_name in sorted(results.metrics.keys()):
        if task_name not in task_results and task_name not in results.groups:
            task_results[task_name] = format_entry(
                task_name, results.metrics[task_name]
            )

    return task_results, group_results


def process_results(
    eval_results_acc: dict[str, ResultAcc],
    groups: dict[str, "Group"] | None = None,
    bootstrap_iters: int | None = 100000,
) -> EvalResults:
    """
    Process evaluation results.

    Args:
        eval_results_acc: Accumulated metrics from evaluation.
            Format: {task_name: {"task": Task, "raw_metrics": defaultdict, "logged_samples": []}}
        groups: Dict of group name -> Group
        bootstrap_iters: Number of bootstrap iterations for stderr calculation

    Returns:
        EvalResults dataclass with:
        - metrics: Dict of task/group metrics
        - configs: Task configurations
        - versions: Task versions
        - num_fewshot: Number of few-shot examples
        - higher_is_better: Metric direction info
        - samples: Sample-level results
        - groups: Groups dict for traversal

    Example usage:
        loaded = task_manager.load_v2(['arc', 'hellaswag'])

        # Run evaluation (populates raw_metrics and logged_samples)
        eval_results_acc = {name: {"task": t, "raw_metrics": defaultdict(list), "logged_samples": []}
                           for name, t in loaded['tasks'].items()}

        results = process_results(eval_results_acc, loaded['groups'])

        # Format for display
        task_results, group_results = format_results(results)
    """
    # Normalize groups to dict
    if groups is None:
        groups = {}

    # Collect task results (includes aggregation)
    results = collect_results(eval_results_acc, groups, bootstrap_iters)

    # Aggregate group metrics
    results = aggregate_groups(results)

    return results


def propagate_higher_is_better(
    all_groups: list["Group"], higher_is_better: dict[str, dict[str, bool]]
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
