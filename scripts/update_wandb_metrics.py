#!/usr/bin/env python3
import wandb
import numpy as np
from tqdm import tqdm
import re
import math
import time

WANDB_PROJECT = "swissai-eval-main-v1-cooldown"
WANDB_ENTITY = "epflmlo-epfl"

wandb.login()

def get_all_runs():
    api = wandb.Api()
    return list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))

def mean(arr):
    return sum(arr) / len(arr)

# NOTE: this is taken from lm_eval, which uses the actual sizes of the subtasks
# but we don't have that here, so we just use 2 for all subtasks (needs to be larger than 1)
def pooled_sample_stderr(stderrs, sizes=None):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    if sizes is None:
        sizes = [2] * len(stderrs)
        print(sizes)

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))

def update_run_metrics(run):
    print(f"Processing run: {run.name}")
    run_id = run.id
    
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
        resume="must"
    ) as update_run:
        
        # Go through each step in the history
        max_step = len(history_list) 
        for step_idx, step in enumerate(tqdm(history_list)):
            if "ConsumedTokens" not in step:
                continue
            # Extract all metrics at this step
            metrics = {k: v for k, v in step.items()}
            
            # Find all hellaswag and arc language-specific metrics
            hellaswag_metrics = {}
            arc_metrics = {}
            
            for metric_name, value in metrics.items():
                if "hellaswag_" in metric_name:
                    metric_type = metric_name.split('/')[-1]  # e.g., "acc", "acc_norm"
                    if metric_type not in hellaswag_metrics:
                        hellaswag_metrics[metric_type] = []
                    hellaswag_metrics[metric_type].append(value)
                
                elif "arc_" in metric_name and not (metric_name.startswith("arc_easy") or metric_name.startswith("arc_challenge")):
                    metric_type = metric_name.split('/')[-1]
                    if metric_type not in arc_metrics:
                        arc_metrics[metric_type] = []
                    arc_metrics[metric_type].append(value)
            
            # Calculate m_hellaswag and m_arc aggregates
            new_metrics = {}
            
            hellaswag_acc_values = hellaswag_metrics.get("acc", [])
            hellaswag_acc_norm_values = hellaswag_metrics.get("acc_norm", [])
            arc_acc_values = arc_metrics.get("acc", [])
            arc_acc_norm_values = arc_metrics.get("acc_norm", [])
            
            # the stderrs
            hellaswag_acc_stderr = hellaswag_metrics.get("acc_stderr", [])
            hellaswag_acc_norm_stderr = hellaswag_metrics.get("acc_norm_stderr", [])
            arc_acc_stderr = arc_metrics.get("acc_stderr", [])
            arc_acc_norm_stderr = arc_metrics.get("acc_norm_stderr", [])

            # Calculate means
            if hellaswag_acc_values:
                new_metrics["m_hellaswag/acc"] = mean(hellaswag_acc_values)
                new_metrics["m_hellaswag/acc_stderr"] = pooled_sample_stderr(hellaswag_acc_stderr)
                
            if hellaswag_acc_norm_values:
                new_metrics["m_hellaswag/acc_norm"] = mean(hellaswag_acc_norm_values)
                new_metrics["m_hellaswag/acc_norm_stderr"] = pooled_sample_stderr(hellaswag_acc_norm_stderr)
                
            if arc_acc_values:
                new_metrics["m_arc/acc"] = mean(arc_acc_values)
                new_metrics["m_arc/acc_stderr"] = pooled_sample_stderr(arc_acc_stderr)
                
            if arc_acc_norm_values:
                new_metrics["m_arc/acc_norm"] = mean(arc_acc_norm_values)
                new_metrics["m_arc/acc_norm_stderr"] = pooled_sample_stderr(arc_acc_norm_stderr)
            
            # Update multilingual metrics
            # Collect task-level accuracy values for proper stderr calculation
            multilingual_acc_values = []
            multilingual_acc_norm_values = []
            multilingual_acc_stderr = []
            multilingual_acc_norm_stderr = []
            
            # Get all task metrics for multilingual benchmarks
            for metric_name, value in metrics.items():
                if '/' in metric_name:
                    task, metric_type = metric_name.split('/')
                    if task in ["global_mmlu", "xcopa", "xnli", "xwinograd", "pawsx"]:
                        if metric_type == "acc":
                            multilingual_acc_values.append(value)
                        elif metric_type == "acc_norm":
                            multilingual_acc_norm_values.append(value)
                        elif metric_type == "acc_stderr":
                            multilingual_acc_stderr.append(value)
                        elif metric_type == "acc_norm_stderr":
                            multilingual_acc_norm_stderr.append(value)
            
            # Add our new aggregates to the multilingual values
            if "m_hellaswag/acc" in new_metrics:
                multilingual_acc_values.append(new_metrics["m_hellaswag/acc"])
            if "m_arc/acc" in new_metrics:
                multilingual_acc_values.append(new_metrics["m_arc/acc"])
            if "m_hellaswag/acc_norm" in new_metrics:
                multilingual_acc_norm_values.append(new_metrics["m_hellaswag/acc_norm"])
            if "m_arc/acc_norm" in new_metrics:
                multilingual_acc_norm_values.append(new_metrics["m_arc/acc_norm"])
            if "m_hellaswag/acc_stderr" in new_metrics:
                multilingual_acc_stderr.append(new_metrics["m_hellaswag/acc_stderr"])
            if "m_arc/acc_stderr" in new_metrics:
                multilingual_acc_stderr.append(new_metrics["m_arc/acc_stderr"])
            if "m_hellaswag/acc_norm_stderr" in new_metrics:
                multilingual_acc_norm_stderr.append(new_metrics["m_hellaswag/acc_norm_stderr"])
            if "m_arc/acc_norm_stderr" in new_metrics:
                multilingual_acc_norm_stderr.append(new_metrics["m_arc/acc_norm_stderr"])
            
            # Calculate multilingual aggregate metrics with proper stderr
            if multilingual_acc_values:
                new_metrics["multilingual/acc"] = mean(multilingual_acc_values)
                new_metrics["multilingual/acc_stderr"] = pooled_sample_stderr(multilingual_acc_stderr)
            
            if multilingual_acc_norm_values:
                new_metrics["multilingual/acc_norm"] = mean(multilingual_acc_norm_values)
                new_metrics["multilingual/acc_norm_stderr"] = pooled_sample_stderr(multilingual_acc_norm_stderr)

            # Update swissai_eval metrics
            # For swissai_eval, we need english and multilingual values
            swissai_eval_acc_values = []
            swissai_eval_acc_norm_values = []
            swissai_eval_acc_stderr = []
            swissai_eval_acc_norm_stderr = []

            # Add english values if they exist
            english_acc = metrics.get("english/acc")
            english_acc_norm = metrics.get("english/acc_norm")
            english_acc_stderr = metrics.get("english/acc_stderr")
            english_acc_norm_stderr = metrics.get("english/acc_norm_stderr")
            if english_acc is not None:
                swissai_eval_acc_values.append(english_acc)
            if english_acc_norm is not None:
                swissai_eval_acc_norm_values.append(english_acc_norm)
            if english_acc_stderr is not None:
                swissai_eval_acc_stderr.append(english_acc_stderr)
            if english_acc_norm_stderr is not None:
                swissai_eval_acc_norm_stderr.append(english_acc_norm_stderr)

            # Add multilingual values if they exist
            multilingual_acc = new_metrics.get("multilingual/acc")
            multilingual_acc_norm = new_metrics.get("multilingual/acc_norm")
            multilingual_acc_stderr = new_metrics.get("multilingual/acc_stderr")
            multilingual_acc_norm_stderr = new_metrics.get("multilingual/acc_norm_stderr")
            
            if multilingual_acc is not None:
                swissai_eval_acc_values.append(multilingual_acc)
            if multilingual_acc_norm is not None:
                swissai_eval_acc_norm_values.append(multilingual_acc_norm)
            if multilingual_acc_stderr is not None:
                swissai_eval_acc_stderr.append(multilingual_acc_stderr)
            if multilingual_acc_norm_stderr is not None:
                swissai_eval_acc_norm_stderr.append(multilingual_acc_norm_stderr)
            
            # Calculate swissai_eval metrics with proper stderr
            if len(swissai_eval_acc_values) > 0:
                new_metrics["swissai_eval/acc"] = mean(swissai_eval_acc_values)
                new_metrics["swissai_eval/acc_stderr"] = pooled_sample_stderr(swissai_eval_acc_stderr)
            
            if len(swissai_eval_acc_norm_values) > 0:
                new_metrics["swissai_eval/acc_norm"] = mean(swissai_eval_acc_norm_values)
                new_metrics["swissai_eval/acc_norm_stderr"] = pooled_sample_stderr(swissai_eval_acc_norm_stderr)
            
                # Add the original metrics to ensure we have everything
                # and overwrite potentially old ones
                all_metrics = {**metrics, **new_metrics}

                # Log the metrics to this step in WandB
                if new_metrics:
                    # Get the step number
                    # step_num = step.get('_step', step_idx)
                    
                    # Log the metrics using our new wandb run
                    wandb.log(all_metrics)
            print(f"Finished processing run: {run.name}")

def main():
    print("Fetching all runs...")
    runs = get_all_runs()
    print(f"Found {len(runs)} runs")
    
    for run in runs:
        update_run_metrics(run)
    
    print("All runs updated successfully!")

if __name__ == "__main__":
    main() 