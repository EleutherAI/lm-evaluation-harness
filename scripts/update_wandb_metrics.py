#!/usr/bin/env python3
import wandb
import numpy as np
from tqdm import tqdm

# Configuration constants
WANDB_PROJECT = "swissai-eval-main-v1.1"
WANDB_ENTITY = "epflmlo-epfl"


def initialize_wandb():
    """Initialize the Weights & Biases connection."""
    wandb.login()


def get_all_runs():
    """Retrieve all runs from the WandB project."""
    api = wandb.Api()
    return list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))


def mean(arr):
    """Calculate the mean of an array."""
    return sum(arr) / len(arr)


MULTILINGUAL_TASKS = [
    "global_mmlu",
    "xcopa",
    "xnli",
    "xwinograd",
    "pawsx",
    "m_hellaswag",
    "m_arc",
]
ENGLISH_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "lambada_openai",
    "lambada_standard",
    "winogrande",
    "piqa",
    "openbookqa",
    "ai2_arc",
    "commonsense_qa",
    "mmlu",
    "mmlu_continuation",
    "gsm8k",
    "wikitext",
    "lambada",
    "hellaswag",
    "squadv2",
]
ENGLISH_METRICS = [
    "acc",
    "acc_norm",
    "f1",
    "perplexity",
    "acc_stderr",
    "acc_norm_stderr",
    "perplexity_stderr",
]
MULTILINGUAL_METRICS = [
    "acc",
    "acc_norm",
    "acc_stderr",
    "acc_norm_stderr",
]

SWISSAI_EVAL_METRICS = [
    "acc",
    "acc_norm",
    "acc_stderr",
    "acc_norm_stderr",
    "perplexity",
    "perplexity_stderr",
    "f1",
]

SWISSAI_EVAL_TASKS = [
    "english_macro",
    "multilingual_macro",
]


# NOTE: this is taken from lm_eval, which uses the actual sizes of the subtasks
# but we don't have that here, so we just use 2 for all subtasks (needs to be larger than 1)
def pooled_sample_stderr(stderrs, sizes=None):
    """
    Aggregate bootstrapped stderrs across subtasks in a group.

    Formula source: https://en.wikipedia.org/wiki/Pooled_variance
    and: https://stats.stackexchange.com/a/4841331
    """
    if sizes is None:
        # this is a hack, we should use the actual sizes of the subtasks, and only use 2 because for 1 there is a divison by zero
        sizes = [2] * len(stderrs)

    assert len(stderrs) == len(sizes)

    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes)).item()


def extract_metrics_by_prefix(metrics, prefix, extract_type=None, ignore_names=None):
    """
    Extract metrics that match a given prefix.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string to match
        extract_type: Whether to extract specific metric types
        ignore_names: List of names to ignore

    Returns:
        Dictionary of metric types with lists of values
    """
    extracted = {}

    for metric_name, value in metrics.items():
        if prefix in metric_name:
            if ignore_names:
                if any(name in metric_name for name in ignore_names):
                    continue
            if extract_type:
                metric_type = metric_name.split("/")[-1]  # e.g., "acc", "acc_norm"
                if metric_type not in extracted:
                    extracted[metric_type] = []
                extracted[metric_type].append(value)
            else:
                extracted[metric_name] = value

    return extracted


def calculate_aggregates(values_dict, prefix):
    """
    Calculate the mean and stderr for a given prefix.

    Args:
        values_dict: Dictionary of lists of values
        prefix: Prefix string to match

    Returns:
        Dictionary of metric types with lists of values
    """
    new_metrics = {}
    for metric_type, values in values_dict.items():
        if values:
            if metric_type.endswith("stderr"):
                new_metrics[f"{prefix}/{metric_type}"] = pooled_sample_stderr(values)
            else:
                new_metrics[f"{prefix}/{metric_type}"] = mean(values)

    return new_metrics


def calculate_aggregates_for_group(metrics, group_name, group_metrics, group_tasks):
    """
    Calculate the mean and stderr for a given group which is defined by a list of tasks and a list of metrics.

    """
    group_metrics = {metric: [] for metric in group_metrics}
    for metric_name, value in metrics.items():
        if "/" in metric_name:
            task, metric_type = metric_name.split("/")
            if task in group_tasks:
                if metric_type in group_metrics:
                    group_metrics[metric_type].append(value)

    group_agg = calculate_aggregates(group_metrics, group_name)
    return group_agg


def process_metrics_for_step(step_metrics):
    if "ConsumedTokens" not in step_metrics:
        return None
    new_metrics = {}
    if "m_hellaswag" not in step_metrics and "m_arc" not in step_metrics:
        # first runs did not include these metrics, compute again
        print("Computing m_hellaswag and m_arc")
        # Extract benchmark-specific metrics
        hellaswag_metrics = extract_metrics_by_prefix(
            step_metrics, "hellaswag_", extract_type=True
        )
        arc_metrics = extract_metrics_by_prefix(
            step_metrics,
            "arc_",
            extract_type=True,
            ignore_names=["arc_easy", "arc_challenge"],
        )

        # Calculate benchmark aggregates
        new_metrics.update(calculate_aggregates(hellaswag_metrics, "m_hellaswag"))
        new_metrics.update(calculate_aggregates(arc_metrics, "m_arc"))
        step_metrics.update(new_metrics)

    # Calculate multilingual aggregates
    multilingual_agg = calculate_aggregates_for_group(
        step_metrics, "multilingual_macro", MULTILINGUAL_METRICS, MULTILINGUAL_TASKS
    )
    new_metrics.update(multilingual_agg)

    # Calculate english aggregates
    english_agg = calculate_aggregates_for_group(
        step_metrics, "english_macro", ENGLISH_METRICS, ENGLISH_TASKS
    )
    new_metrics.update(english_agg)

    # Calculate swissai_eval aggregates with the new metrics
    swissai_agg = calculate_aggregates_for_group(
        new_metrics, "swissai_eval_macro", SWISSAI_EVAL_METRICS, SWISSAI_EVAL_TASKS
    )
    new_metrics.update(swissai_agg)
    new_metrics["ConsumedTokens"] = step_metrics["ConsumedTokens"]

    return new_metrics


def update_run_metrics(run):
    print(f"Processing run: {run.name}")

    history = run.scan_history()
    history_list = list(history)
    if not history_list:
        print(f"No history found for run {run.name}")
        return

    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=run.id,
        config=run.config,
        resume="must",
    ) as update_run:
        # Process each step in the history
        for step in tqdm(history_list):
            updated_metrics = process_metrics_for_step(step)
            if updated_metrics:
                wandb.log(updated_metrics)

    print(f"Finished processing run: {run.name}")


def main():
    initialize_wandb()

    print("Fetching all runs...")
    runs = get_all_runs()
    print(f"Found {len(runs)} runs")

    for run in runs:
        update_run_metrics(run)

    print("All runs updated successfully!")


if __name__ == "__main__":
    main()
