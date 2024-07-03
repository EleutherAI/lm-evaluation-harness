import collections
import math
import pathlib
import sys
from typing import Dict, List, Optional, Tuple, Union

from lm_eval.api import metrics
from lm_eval.utils import eval_logger, positional_deprecated


class TaskOutput:
    """
    Wrapper class for Task outputs.It contains various attributes and methods to manage and calculate metrics for the task.

        Attributes:
            task (object): The task object.
            task_name (str): The name of the task.
            task_config (dict): The configuration of the task.
            version (str): The version of the task.
            group_name (str): The name of the task group.
            n_shot (int): The number of shots for the task.
            task_alias (str): The alias of the task.
            group_alias (str): The alias of the task group.
            is_group (bool): Indicates if the task is a group.
            logged_samples (list): The list of logged samples.
            sample_len (int): The length of the samples.
            sample_metrics (defaultdict): The dictionary of samples' metrics.
            agg_metrics (defaultdict): The dictionary of aggregate metrics.

        Methods:
            from_taskdict(cls, task_name: str, task):
                Creates a TaskOutput instance from a task dictionary.

            calculate_aggregate_metric(bootstrap_iters=100000) -> None:
                Calculates the aggregate metrics for the task.
    """

    def __init__(
        self,
        task=None,
        task_name=None,
        task_config=None,
        version=None,
        group_name=None,
        n_shot=None,
        task_alias=None,
        group_alias=None,
        is_group=None,
    ):
        self.task = task
        self.task_config = task_config
        self.task_name = task_name
        self.group_name = group_name
        self.version = version
        self.n_shot = n_shot
        self.task_alias = task_alias
        self.group_alias = group_alias
        self.is_group = is_group
        self.logged_samples = []
        self.sample_len = None
        self.sample_metrics = collections.defaultdict(list)
        self.agg_metrics = collections.defaultdict(list)

    @classmethod
    def from_taskdict(cls, task_name: str, task):
        if isinstance(task, tuple):
            group_name, task = task
        else:
            group_name = None
        if not task:
            # these gets filtered out in get_task_list
            # once they are added to group hierarchy
            is_group = True
            return cls(
                task=task, task_name=task_name, is_group=is_group, group_name=group_name
            )
        version = task.VERSION
        task_config = dict(task.dump_config())
        if (n_shot := task_config.get("num_fewshot")) == 0:
            n_shot = task_config.get("metadata", {}).get("num_fewshot", 0)
        task_alias = task_config.get("alias")
        group_alias = task_config.get("group_alias")
        return cls(
            task=task,
            task_name=task_name,
            task_config=task_config,
            group_name=group_name,
            version=version,
            n_shot=n_shot,
            task_alias=task_alias,
            group_alias=group_alias,
        )

    def calculate_aggregate_metric(self, bootstrap_iters=100000) -> None:
        for (metric, filter_key), items in self.sample_metrics.items():
            agg_fn = self.task.aggregation()[metric]
            metric_key = f"{metric},{filter_key}"
            self.agg_metrics[metric_key] = agg_fn(items)
            self.sample_len = len(items)  # TODO: same sample size for each metric?
            if isinstance(bootstrap_iters, int):
                stderr_fn = metrics.stderr_for_metric(
                    metric=agg_fn,
                    bootstrap_iters=min(bootstrap_iters, 100)
                    if metric in ["bleu", "chrf", "ter"]
                    else bootstrap_iters,
                )
                self.agg_metrics[f"{metric}_stderr,{filter_key}"] = (
                    stderr_fn(items) if (stderr_fn and len(items) > 1) else "N/A"
                )
            else:
                raise ValueError(
                    f"Received bootstrap_iters '{bootstrap_iters}' but expected an integer. Set to 0 to turn off stderr calculations."
                )

    def __repr__(self):
        return (
            f"TaskOutput(task_name={self.task_name}, "
            f"group_name={self.group_name}, "
            f"version={self.version},"
            f"n_shot={self.n_shot}"
            f"task_alias={self.task_alias}, group_alias={self.group_alias})"
        )


def get_task_list(task_dict: dict) -> Tuple[Dict[str, list], List[TaskOutput]]:
    task_hierarchy = collections.defaultdict(list)
    outputs = list(TaskOutput.from_taskdict(x, y) for x, y in task_dict.items())
    for task_output in outputs:
        if group_name := task_output.group_name:
            task_hierarchy[group_name].append(task_output.task_name)
        else:
            task_hierarchy[task_output.task_name] = []
    # returns task_hierarchy tracking which groups contain which subtasks,
    # and a list of TaskOutput classes for each non-group subtask
    return task_hierarchy, [x for x in outputs if x.task]


def print_writeout(task) -> None:
    for inst in task.instances:
        # print the prompt for the first few documents
        if inst.doc_id < 1:
            eval_logger.info(
                f"Task: {task}; document {inst.doc_id}; context prompt (starting on next line):\
    \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
            )
            eval_logger.info(f"Request: {str(inst)}")


def get_sample_size(task, limit: Optional[int]) -> Union[int, None]:
    if limit is not None:
        limit = (
            int(math.ceil(len(task.eval_docs) * limit)) if limit < 1.0 else int(limit)
        )
    return limit


def prepare_print_tasks(
    task_hierarchy: dict, results: dict, tab=0
) -> Tuple[dict, dict]:
    """
    @param task_hierarchy: Dictionary representing the group hierarchy of tasks. Each key is a group name and its
    value is a list of task names.
    @param results: Dictionary containing the results of each task. Each key is a
    group name and its value is a dictionary of task results.
    @param tab: The indentation level for printing the task
    hierarchy. Default is 0.
    @return: A tuple of two dictionaries: results_agg and groups_agg. results_agg contains
    aggregated results for each task, and groups_agg contains aggregated results for each group.

    Prepares the task hierarchy and aggregates the results for each task and group recursively for printing.
    """
    results_agg = collections.defaultdict(dict)
    groups_agg = collections.defaultdict(dict)

    (group_name, task_list), *_ = task_hierarchy.items()
    task_list = sorted(task_list)

    results_agg[group_name] = results[group_name].copy()
    # results_agg[group_name]["tab"] = tab
    if "samples" in results_agg[group_name]:
        results_agg[group_name].pop("samples")

    tab_string = " " * tab + "- " if tab > 0 else ""

    if "alias" in results_agg[group_name]:
        results_agg[group_name]["alias"] = tab_string + results_agg[group_name]["alias"]
    else:
        results_agg[group_name]["alias"] = tab_string + group_name

    if len(task_list) > 0:
        groups_agg[group_name] = results[group_name].copy()
        # groups_agg[group_name]["tab"] = tab
        if "samples" in groups_agg[group_name]:
            groups_agg[group_name].pop("samples")

        if "alias" in groups_agg[group_name]:
            groups_agg[group_name]["alias"] = (
                tab_string + groups_agg[group_name]["alias"]
            )
        else:
            groups_agg[group_name]["alias"] = tab_string + group_name

        for task_name in task_list:
            if task_name in task_hierarchy:
                _task_hierarchy = {
                    **{task_name: task_hierarchy[task_name]},
                    **task_hierarchy,
                }
            else:
                _task_hierarchy = {
                    **{task_name: []},
                    **task_hierarchy,
                }

            _results_agg, _groups_agg = prepare_print_tasks(
                _task_hierarchy, results, tab + 1
            )
            results_agg = {**results_agg, **_results_agg}
            groups_agg = {**groups_agg, **_groups_agg}

    return results_agg, groups_agg


def consolidate_results(
    eval_tasks: List[TaskOutput],
) -> Tuple[dict, dict, dict, dict, dict, dict]:
    """
    @param eval_tasks: list(TaskOutput).
    @return: A tuple containing the consolidated results, samples, configs, versions, and num_fewshot.

    Consolidates the results of multiple evaluation tasks into a single structure.

    The method iterates over each evaluation instance and extracts relevant information to create the consolidated
    results structure. The consolidated results structure has the following properties:

    - results: A defaultdict with task names as keys and dictionaries as values. Each dictionary contains
    metric/filter pairs as keys and corresponding metric values as values. The "alias" key is used to store task
    aliases specified in the task configuration.
    - samples: A defaultdict with task names as keys and lists of log samples as values.
    - configs: A defaultdict with task names as keys and task configurations as values.
    - versions: A defaultdict with task names as keys and task versions as values.
    - num_fewshot: A defaultdict with task names as keys and number of few-shot samples as values.
    - higher_is_better: A defaultdict with task names as keys and indicators of whether higher values are better
    for each metric as values.

    The method then returns the consolidated results, samples, configs, versions, and num_fewshot as a tuple.
    """
    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    # Tracks the YAML configs of all chosen task
    configs = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Track `higher_is_better` for each metric
    higher_is_better = collections.defaultdict(dict)

    for task_output in eval_tasks:
        if "task_alias" in (task_config := task_output.task_config):
            results[task_output.task_name]["alias"] = task_config["task_alias"]
        if group_alias := task_output.group_alias:
            if group_alias not in results and (group_name := task_output.group_name):
                results[group_name]["alias"] = group_alias
        num_fewshot[task_output.task_name] = task_output.n_shot
        configs[task_output.task_name] = task_output.task_config
        versions[task_output.task_name] = task_output.version
        samples[task_output.task_name] = task_output.logged_samples
        higher_is_better[task_output.task_name] = task_output.task.higher_is_better()
        for (metric, filter_key), items in task_output.sample_metrics.items():
            metric_key = f"{metric},{filter_key}"
            results[task_output.task_name][metric_key] = task_output.agg_metrics[
                metric_key
            ]
            results[task_output.task_name]["samples"] = task_output.sample_len
            results[task_output.task_name][f"{metric}_stderr,{filter_key}"] = (
                task_output.agg_metrics[f"{metric}_stderr,{filter_key}"]
            )
    return results, samples, configs, versions, num_fewshot, higher_is_better


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
def run_task_tests(task_list: List[str]):
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
