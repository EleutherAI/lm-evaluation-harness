#!/usr/bin/env python3
"""
Script to compare evaluation results across different models for nl2foam tasks.
Creates comparison tables for all tasks found in the results files.

Usage:
    python compare_results.py [directory]
    
    If directory is not specified, defaults to "all-results".
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional


def get_model_name(config: Dict[str, Any]) -> str:
    """Extract a readable model name from the config."""
    pretrained = config.get("model_args", {}).get("pretrained", "")
    if not pretrained:
        return "Unknown"
    
    # Use the last part of the path or the full name if it's a short identifier
    if "/" in pretrained:
        parts = pretrained.split("/")
        # If it's a path (starts with /), use the last meaningful part
        if pretrained.startswith("/"):
            # Extract the model name from path like /scratch/.../nl2foam_sft_0__8__1765350318
            return parts[-1] if parts[-1] else parts[-2]
        else:
            # For HuggingFace models like "YYgroup/AutoCFD-7B"
            return pretrained
    return pretrained


def load_results_file(filepath: Path) -> Dict[str, Any]:
    """Load a results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(task_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract metrics and their stderr values from task results.
    Returns a dict mapping metric_name -> {'value': ..., 'stderr': ...}
    """
    metrics = {}
    
    for key, value in task_results.items():
        # Skip alias and other non-metric keys
        if key == "alias":
            continue
        
        # Check if this is a stderr key
        if key.endswith("_stderr,none"):
            metric_name = key.replace("_stderr,none", "")
            if metric_name not in metrics:
                metrics[metric_name] = {}
            metrics[metric_name]["stderr"] = value
        elif key.endswith(",none"):
            metric_name = key.replace(",none", "")
            if metric_name not in metrics:
                metrics[metric_name] = {}
            metrics[metric_name]["value"] = value
    
    return metrics


def format_value(value: Any, stderr: Optional[Any] = None) -> str:
    """Format a metric value with optional stderr."""
    if value is None:
        return "N/A"
    
    if isinstance(value, str) and value == "N/A":
        return "N/A"
    
    if isinstance(value, (int, float)):
        if stderr is not None and stderr != "N/A" and isinstance(stderr, (int, float)):
            return f"{value:.6f} ± {stderr:.6f}"
        else:
            return f"{value:.6f}"
    
    return str(value)


def create_comparison_table(
    task_name: str,
    model_data: Dict[str, Dict[str, Dict[str, Any]]]
) -> str:
    """
    Create a markdown-formatted comparison table for a task.
    
    Args:
        task_name: Name of the task
        model_data: Dict mapping model_name -> metric_name -> {'value': ..., 'stderr': ...}
    """
    if not model_data:
        return f"\n## {task_name}\n\nNo data available.\n\n"
    
    # Get all unique metrics across all models
    all_metrics = set()
    for metrics in model_data.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    
    if not all_metrics:
        return f"\n## {task_name}\n\nNo metrics found.\n\n"
    
    # Get all model names
    model_names = sorted(model_data.keys())
    
    # Build markdown table
    lines = []
    lines.append(f"\n## {task_name}\n")
    
    # Header row
    header = "| Metric |"
    separator = "|--------|"
    for model_name in model_names:
        header += f" {model_name} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)
    
    # Data rows
    for metric in all_metrics:
        row = f"| {metric} |"
        for model_name in model_names:
            metric_data = model_data[model_name].get(metric, {})
            value = metric_data.get("value")
            stderr = metric_data.get("stderr")
            formatted = format_value(value, stderr)
            # Escape pipes in the formatted value
            formatted = formatted.replace("|", "\\|")
            row += f" {formatted} |"
        lines.append(row)
    
    lines.append("")
    return "\n".join(lines)


def main():
    """Main function to process all results files and create comparison tables."""
    # Get directory from command line argument or use default
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("all-results")
    
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' not found")
        return
    
    # Find all results JSON files
    results_files = sorted(results_dir.glob("results_*.json"))
    
    if not results_files:
        print(f"Error: No results_*.json files found in '{results_dir}'")
        return
    
    print(f"Found {len(results_files)} results files in '{results_dir}'")
    
    # Organize data by task and model
    # Structure: task_name -> model_name -> metric_name -> {'value': ..., 'stderr': ...}
    task_data = defaultdict(lambda: defaultdict(dict))
    all_tasks = set()
    
    # Process each results file
    for filepath in results_files:
        try:
            data = load_results_file(filepath)
            results = data.get("results", {})
            config = data.get("config", {})
            
            model_name = get_model_name(config)
            
            # Process each task in this file
            for task_name, task_results in results.items():
                all_tasks.add(task_name)
                metrics = extract_metrics(task_results)
                task_data[task_name][model_name] = metrics
                
        except Exception as e:
            print(f"Warning: Error processing {filepath}: {e}")
            continue
    
    if not task_data:
        print("Error: No task data found in any results files")
        return
    
    # Create markdown output
    output_lines = []
    output_lines.append("# Model Comparison Results\n")
    output_lines.append(f"Comparison of evaluation results across different models.\n")
    output_lines.append(f"Source directory: `{results_dir}`\n")
    
    # Create comparison tables for each task (sorted for consistent output)
    tasks = sorted(all_tasks)
    
    for task in tasks:
        if task in task_data:
            table = create_comparison_table(task, task_data[task])
            output_lines.append(table)
        else:
            output_lines.append(f"\n## {task}\n\nNo data available.\n\n")
    
    # Write to markdown file
    output_file = results_dir / "model_comparison.md"
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))
    
    print(f"Comparison tables saved to: {output_file}")
    print(f"Found {len(tasks)} task(s): {', '.join(tasks)}")


if __name__ == "__main__":
    main()

