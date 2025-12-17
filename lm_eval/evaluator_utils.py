import collections
import logging
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict

from lm_eval.api.group import ConfigurableGroup, Group
from lm_eval.api.metrics import (
    aggregate_subtask_metrics,
    mean,
    pooled_sample_stderr,
    stderr_for_metric,
)
from lm_eval.api.task import Task
from lm_eval.utils import positional_deprecated


eval_logger = logging.getLogger(__name__)


class ResultAcc(TypedDict):
    """Accumulator for results of a single task."""

    task: Task
    raw_metrics: dict
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


def get_sample_size(task, limit: int | None) -> int | None:
    if limit is not None:
        limit = (
            int(math.ceil(len(task.eval_docs) * limit)) if limit < 1.0 else int(limit)
        )
    return limit


def prepare_print_tasks(
    task_dict: dict,
    results: dict,
    task_depth=0,
    group_depth=0,
) -> tuple[dict, dict]:
    """
    @param task_dict: Dictionary representing the group hierarchy of tasks. Each key is a group name and its
    value is a list of task names.
    @param results: Dictionary containing the results of each task. Each key is a
    group name and its value is a dictionary of task results.
    @param task_depth: The indentation level for printing the task
    hierarchy. Default is 0.
    @param group_depth: The indentation level for printing the group
    hierarchy. Default is 0.
    @return: A tuple of two dictionaries: results_agg and groups_agg. results_agg contains
    aggregated results for each task, and groups_agg contains aggregated results for each group.

    Prepares the task hierarchy and aggregates the results for each task and group recursively for printing.
    """

    def _sort_task_dict(task_dict):
        """
        Helper utility. Sorts the task dict at the current level of the hierarchy based on alphabetized task name.
        Required so that we end up sorting within each sub-header correctly.
        """

        return dict(
            sorted(
                task_dict.items(),
                key=lambda item: item[0].group_name
                if isinstance(item[0], ConfigurableGroup)
                else item[0],
            )
        )

    task_agg = collections.defaultdict(dict)
    group_agg = collections.defaultdict(dict)
    task_dict = _sort_task_dict(task_dict)
    for task_or_group_name, task_or_group_obj in task_dict.items():
        tab_string = " " * task_depth + "- " if task_depth > 0 else ""
        if isinstance(task_or_group_name, ConfigurableGroup):
            # string_name = task_or_group_name.group_name
            name = task_or_group_name.group_name
            from_configurable_group = True
            task_or_group_obj = _sort_task_dict(task_or_group_obj)
        elif isinstance(task_or_group_name, str):
            # Use the dict key as the result lookup key (preserves hierarchical child names)
            name = task_or_group_name
            from_configurable_group = False

        task_agg[name] = results[name].copy()
        if from_configurable_group:
            if task_or_group_name.group_alias is not None:
                alias = task_or_group_name.group_alias
            else:
                alias = task_or_group_name.group
        else:
            alias = task_agg[name].get("alias", name)

        task_agg[name]["alias"] = tab_string + alias
        if "samples" in task_agg[name]:
            task_agg[name].pop("samples")

        if from_configurable_group and (" " not in results[name]):
            group_tab_string = " " * group_depth + "- " if group_depth > 0 else ""
            group_agg[name] = results[name].copy()
            group_agg[name]["alias"] = group_tab_string + alias
            if "samples" in group_agg[name]:
                group_agg[name].pop("samples")

        if isinstance(task_or_group_obj, dict):
            task_depth += 1
            group_depth += 1
            _task_agg, _group_agg = prepare_print_tasks(
                task_or_group_obj, results, task_depth, group_depth
            )
            task_agg = {
                **task_agg,
                **_task_agg,
            }
            group_agg = {**group_agg, **_group_agg}
            task_depth -= 1
            group_depth -= 1
    return task_agg, group_agg


def consolidate_group_results(
    results,
    versions,
    task_dict,
    task_root=None,
    show_group_table=False,
    task_aggregation_list=None,
) -> tuple[dict, dict, bool, None]:
    """
    (Recursively) calculates groups' aggregated metrics and updates the results and versions dictionaries with this info.

    @return: a tuple [results, versions, show_group_table, task_aggregation_list] with formats described below:

    - results: A defaultdict with task names (and, after this function is called, group names of
    groups that perform aggregation) as keys, and dictionaries with "alias" and metric,filter_name pairs as keys.
    - versions: A defaultdict with task names (and, after this function is called, group names of
    groups that perform aggregation) as keys, and float values representing the task or group's version if a version is specified. (defaulting to None).
    - show_group_table: a boolean which is true if there exists a group that requires printing of its aggregated scores in a group table.
    - task_aggregation_list: a defaultdict listing the subtasks to average over to produce a given group's end metric.

    The method then returns the updated results, versions, show_group_table, and task_aggregation_list as a tuple.
    In the top-level invocation of this function, task_aggregation_list is ignored.
    """
    if task_root is None:
        task_root = {}

    if task_aggregation_list is None:
        task_aggregation_list = {}

    for group_or_task, group_or_task_info in task_dict.items():
        # Convert to string
        if isinstance(group_or_task, ConfigurableGroup):
            group_config = group_or_task.config
            group_or_task = group_or_task.group_name
        else:
            group_config = None

        if isinstance(group_or_task_info, Task):
            if task_root:
                task_aggregation_list.setdefault(task_root, []).append(
                    group_or_task_info.task_name
                )
        else:
            (
                results,
                versions,
                show_group_table,
                _task_aggregation_list,
            ) = consolidate_group_results(
                results,
                versions,
                group_or_task_info,
                group_or_task,
                show_group_table,
                task_aggregation_list,
            )
            if task_root:
                task_aggregation_list.setdefault(task_root, []).extend(
                    task_aggregation_list.get(group_or_task, [])
                )

            if (group_config is None) or (
                group_config["aggregate_metric_list"] is None
            ):
                results[group_or_task][" "] = " "
                continue

            if "aggregate_metric_list" in group_config:
                agg_metric_list = group_config["aggregate_metric_list"]

            show_group_table = show_group_table | bool(
                group_config["aggregate_metric_list"]
            )

            task_list = _task_aggregation_list[group_or_task]

            metric_list = list(
                {
                    key
                    for task in task_list
                    for key in results[task].keys()
                    if "_stderr" not in key and key not in ["task", "alias", "samples"]
                }
            )
            for metric in metric_list:
                stderr = "_stderr,".join(metric.split(","))

                # gather metrics, sizes, and stderrs from subtasks
                metrics = [
                    results[task][metric]
                    for task in task_list
                    if metric in results[task]
                ]  # TODO: copy?
                stderrs = [
                    results[task][stderr]
                    for task in task_list
                    if stderr in results[task]
                ]
                sizes = [
                    results[task]["samples"]
                    for task in task_list
                    if metric in results[task]
                ]

                for metric_config in agg_metric_list:
                    for filter_name in metric_config["filter_list"]:
                        if metric != ",".join([metric_config["metric"], filter_name]):
                            continue

                        # compute group's pooled metric and stderr
                        if metric_config["aggregation"] == "mean":
                            aggregate_fn = aggregate_subtask_metrics
                        elif callable(metric_config["aggregation"]):
                            aggregate_fn = metric_config["aggregation"]
                        else:
                            raise ValueError(
                                f"Currently, only 'mean' is supported for automatically aggregating scores across groups' subtasks. Got '{metric_config['aggregation']}' for group '{group_or_task}'"
                            )

                        results[group_or_task][metric] = aggregate_fn(
                            metrics,
                            sizes,
                            metric_config["weight_by_size"],
                        )
                        # TODO: calculate groups' metrics using arbitrary agg fns
                        if "N/A" in stderrs:
                            results[group_or_task][stderr] = "N/A"
                        else:
                            # NOTE: this assumes we are using the mean to aggregate. There are warnings about this elsewhere
                            results[group_or_task][stderr] = pooled_sample_stderr(
                                stderrs, sizes
                            )

                results[group_or_task]["samples"] = sum(sizes)
                group_metadata = group_config.get("metadata", None)
                if group_metadata is not None:
                    versions[group_or_task] = group_metadata.get("version", None)

            # Clean up duplicate score rows for subtasks that also report other metrics.
            for task in task_list:
                task_metrics = [
                    key
                    for key in results[task].keys()
                    if "," in key and not key.startswith("score_stderr")
                ]
                score_metrics = [
                    key for key in task_metrics if key.startswith("score,")
                ]
                if score_metrics and len(task_metrics) > len(score_metrics):
                    for score_metric in score_metrics:
                        results[task].pop(score_metric, None)
                        stderr_key = score_metric.replace("score,", "score_stderr,")
                        results[task].pop(stderr_key, None)
    # print(results)
    return results, versions, show_group_table, task_aggregation_list


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


def compute_task_metrics(
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
            agg_fn = task.aggregation()[metric]
        except KeyError:
            # Arbitrary metric without a defined aggregation function
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
        agg_metrics, sample_len = compute_task_metrics(
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

    return results


def get_root_groups(groups: dict[str, "Group"]) -> list["Group"]:
    """
    Find groups that aren't children of any other group.

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
        for child in group.children:
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
