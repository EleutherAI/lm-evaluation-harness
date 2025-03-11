#!/usr/bin/env python3
import wandb
import numpy as np
from tqdm import tqdm
import re
import time

WANDB_PROJECT = "swissai-eval-main-v1"
WANDB_ENTITY = "epflmlo-epfl"

wandb.login()

def get_all_runs():
    api = wandb.Api()
    return list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))

def calculate_stderr(values):
    if len(values) <= 1:
        return 0
    return np.std(values, ddof=1) / np.sqrt(len(values))

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
            # Extract all metrics at this step
            metrics = {k: v for k, v in step.items()}
            
            # Find all hellaswag and arc language-specific metrics
            hellaswag_metrics = {}
            arc_metrics = {}
            
            for metric_name, value in metrics.items():
                if "hellaswag_" in metric_name and not metric_name.startswith("hellaswag_multilingual"):
                    metric_type = metric_name.split('/')[-1]  # e.g., "acc", "acc_norm"
                    if metric_type not in hellaswag_metrics:
                        hellaswag_metrics[metric_type] = []
                    hellaswag_metrics[metric_type].append(value)
                
                elif "arc_" in metric_name and not metric_name.startswith("arc_multilingual"):
                    metric_type = metric_name.split('/')[-1]
                    if metric_type not in arc_metrics:
                        arc_metrics[metric_type] = []
                    arc_metrics[metric_type].append(value)
            
            # Calculate m_hellaswag and m_arc aggregates
            new_metrics = {}
            
            # Store raw accuracy values for proper stderr calculation
            hellaswag_acc_values = hellaswag_metrics.get("acc", [])
            hellaswag_acc_norm_values = hellaswag_metrics.get("acc_norm", [])
            arc_acc_values = arc_metrics.get("acc", [])
            arc_acc_norm_values = arc_metrics.get("acc_norm", [])
            
            # Calculate means
            if hellaswag_acc_values:
                new_metrics["m_hellaswag/acc"] = np.mean(hellaswag_acc_values)
                new_metrics["m_hellaswag/acc_stderr"] = calculate_stderr(hellaswag_acc_values)
                
            if hellaswag_acc_norm_values:
                new_metrics["m_hellaswag/acc_norm"] = np.mean(hellaswag_acc_norm_values)
                new_metrics["m_hellaswag/acc_norm_stderr"] = calculate_stderr(hellaswag_acc_norm_values)
                
            if arc_acc_values:
                new_metrics["m_arc/acc"] = np.mean(arc_acc_values)
                new_metrics["m_arc/acc_stderr"] = calculate_stderr(arc_acc_values)
                
            if arc_acc_norm_values:
                new_metrics["m_arc/acc_norm"] = np.mean(arc_acc_norm_values)
                new_metrics["m_arc/acc_norm_stderr"] = calculate_stderr(arc_acc_norm_values)
            
            # Update multilingual metrics
            # Collect task-level accuracy values for proper stderr calculation
            multilingual_acc_values = []
            multilingual_acc_norm_values = []
            
            # Get all task metrics for multilingual benchmarks
            for metric_name, value in metrics.items():
                if '/' in metric_name:
                    task, metric_type = metric_name.split('/')
                    if task in ["global_mmlu", "xcopa", "xnli", "xwinograd", "pawsx"]:
                        if metric_type == "acc":
                            multilingual_acc_values.append(value)
                        elif metric_type == "acc_norm":
                            multilingual_acc_norm_values.append(value)
            
            # Add our new aggregates to the multilingual values
            if "m_hellaswag/acc" in new_metrics:
                multilingual_acc_values.append(new_metrics["m_hellaswag/acc"])
            if "m_arc/acc" in new_metrics:
                multilingual_acc_values.append(new_metrics["m_arc/acc"])
            if "m_hellaswag/acc_norm" in new_metrics:
                multilingual_acc_norm_values.append(new_metrics["m_hellaswag/acc_norm"])
            if "m_arc/acc_norm" in new_metrics:
                multilingual_acc_norm_values.append(new_metrics["m_arc/acc_norm"])
            
            # Calculate multilingual aggregate metrics with proper stderr
            if multilingual_acc_values:
                new_metrics["multilingual/acc"] = np.mean(multilingual_acc_values)
                new_metrics["multilingual/acc_stderr"] = calculate_stderr(multilingual_acc_values)
            
            if multilingual_acc_norm_values:
                new_metrics["multilingual/acc_norm"] = np.mean(multilingual_acc_norm_values)
                new_metrics["multilingual/acc_norm_stderr"] = calculate_stderr(multilingual_acc_norm_values)
  
            # Update swissai_eval metrics
            # For swissai_eval, we need english and multilingual values
            swissai_eval_acc_values = []
            swissai_eval_acc_norm_values = []
            
            # Add english values if they exist
            english_acc = metrics.get("english/acc")
            english_acc_norm = metrics.get("english/acc_norm")
            
            if english_acc is not None:
                swissai_eval_acc_values.append(english_acc)
            if english_acc_norm is not None:
                swissai_eval_acc_norm_values.append(english_acc_norm)
            
            # Add multilingual values if they exist
            multilingual_acc = new_metrics.get("multilingual/acc")
            multilingual_acc_norm = new_metrics.get("multilingual/acc_norm")
            
            if multilingual_acc is not None:
                swissai_eval_acc_values.append(multilingual_acc)
            if multilingual_acc_norm is not None:
                swissai_eval_acc_norm_values.append(multilingual_acc_norm)
            
            # Calculate swissai_eval metrics with proper stderr
            if len(swissai_eval_acc_values) > 0:
                new_metrics["swissai_eval/acc"] = np.mean(swissai_eval_acc_values)
                new_metrics["swissai_eval/acc_stderr"] = calculate_stderr(swissai_eval_acc_values)
            
            if len(swissai_eval_acc_norm_values) > 0:
                new_metrics["swissai_eval/acc_norm"] = np.mean(swissai_eval_acc_norm_values)
                new_metrics["swissai_eval/acc_norm_stderr"] = calculate_stderr(swissai_eval_acc_norm_values)
            
            # Add the original metrics to ensure we have everything
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