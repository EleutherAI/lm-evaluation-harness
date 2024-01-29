import copy
import re

from packaging.version import Version


IS_WANDB_AVAILABLE = False

try:
    import wandb

    assert Version(wandb.__version__) >= Version("0.13.6")
    if Version(wandb.__version__) < Version("0.13.6"):
        wandb.require("report-editing:v0")
    IS_WANDB_AVAILABLE = True
except:
    logger.warning(
        "To use the wandb reporting functionality please install wandb>=0.13.6.\n"
        "To install the latest version of wandb run `pip install wandb --upgrade`"
    )
    IS_WANDB_AVAILABLE = False

if IS_WANDB_AVAILABLE:
    pass


def remove_none_pattern(input_string):
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def get_config(results):
    task_configs = results.get("configs", {})
    cli_configs = results.get("config", {})
    configs = {
        "task_configs": task_configs,
        "cli_configs": cli_configs,
    }

    return configs


def sanitize_results_dict(results):
    """
    Remove string valued keys from the results dict as they don't render in the workspace.
    Log these key-value pairs to wandb.summary.
    """
    _results = results.get("results", dict())
    task_names = list(_results.keys())

    # Remove None from the metric string name
    tmp_results = copy.deepcopy(_results)
    for task in task_names:
        task_result = tmp_results.get(task, dict())
        for metric_name, metric_value in task_result.items():
            _metric_name, removed = remove_none_pattern(metric_name)
            if removed:
                _results[task][_metric_name] = metric_value
                _results[task].pop(metric_name)

    # remove string valued keys from the results dict
    wandb_summary = {}
    for task in task_names:
        task_result = _results.get(task, dict())
        for metric_name, metric_value in task_result.items():
            if isinstance(metric_value, str):
                wandb_summary[f"{task}/{metric_name}"] = metric_value

    for summary_metric, summary_value in wandb_summary.items():
        _task, _summary_metric = summary_metric.split("/")
        _results[_task].pop(_summary_metric)

    tmp_results = copy.deepcopy(_results)
    for task_name, task_results in tmp_results.items():
        for metric_name, metric_value in task_results.items():
            _results[f"{task_name}/{metric_name}"] = metric_value
            _results[task_name].pop(metric_name)
    for task in task_names:
        _results.pop(task)

    return wandb_summary, _results


def log_eval_result(wandb_args_dict, results):
    # initialize a W&B run
    run = wandb.init(**wandb_args_dict)

    # Log configs to wandb
    configs = get_config(results)
    run.config.update(configs)

    wandb_summary, wandb_results = sanitize_results_dict(results)
    # update wandb.run.summary with items that were removed
    run.summary.update(wandb_summary)
    # Log the evaluation metrics to wandb
    wandb.log(wandb_results)


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d (dict): The nested dictionary to be flattened.
    - parent_key (str, optional): The key from the parent dictionary (used for recursion). Defaults to an empty string.
    - sep (str, optional): The separator used between keys when generating the flattened keys. Defaults to '_'.

    Returns:
    - dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def remove_keys_with_substrings(d, substrings_to_remove):
    """
    Remove keys containing specified substrings from a dictionary.

    Parameters:
    - d (dict): The original dictionary.
    - substrings_to_remove (list): List of substrings to be removed from keys.

    Returns:
    - dict: The modified dictionary.
    """
    return {key: value for key, value in d.items() if not any(substring in key for substring in substrings_to_remove)}


def log_eval_samples(log_samples):
    assert wandb.run is not None

    task_names = list(log_samples.keys())
    for task_name in task_names:
        eval_preds = log_samples[task_name]

        _eval_preds = []
        for eval_pred in eval_preds:
            eval_pred = flatten_dict(eval_pred)
            eval_pred = remove_keys_with_substrings(
                eval_pred,
                substrings_to_remove=['resps',],
            )
            _eval_preds.append(eval_pred)

        # initialize a new W&B Table
        columns = list(_eval_preds[0].keys())
        table = wandb.Table(columns=columns)
        # TODO: handle columns with list data in a better way
        for _eval_pred in _eval_preds:
            table.add_data(*_eval_pred.values())

        # log the table to W&B
        wandb.run.log({f"{task_name}_eval_results": table})

        del _eval_preds
