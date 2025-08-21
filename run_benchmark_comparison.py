#!/usr/bin/env python3
"""
Script to run and compare long-context benchmarks with official implementations
"""

import json
import os
import subprocess
from datetime import datetime


def run_benchmark(model_name, tasks, output_dir):
    """Run a benchmark and save results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/{tasks}_{timestamp}"

    cmd = [
        "python3",
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_name},trust_remote_code=True",
        "--tasks",
        tasks,
        "--batch_size",
        "1",
        "--output_path",
        output_path,
        "--limit",
        "5",  # Using small limit for demonstration
        "--verbosity",
        "INFO",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Output will be saved to: {output_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✓ {tasks} completed successfully")
            # Try to read and display results
            results_file = f"{output_path}/results.json"
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                    return results
        else:
            print(f"✗ {tasks} failed:")
            print(result.stderr[:500])
    except subprocess.TimeoutExpired:
        print(f"✗ {tasks} timed out")
    except Exception as e:
        print(f"✗ {tasks} error: {e}")

    return None


def main():
    # Create output directory
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)

    # Model to test (using small model for demonstration)
    model = "openai-community/gpt2"  # Small model for quick testing

    # Benchmarks to run
    benchmarks = {
        "longbench_v2": ["2wikimqa", "book_qa_eng"],  # Sample tasks
        "babilong": ["qa1_single_fact", "qa2_two_supporting_facts"],  # Sample tasks
        "infinitebench": ["kv_retrieval"],  # Simple task
        "phonebook": ["phonebook_1k"],  # Smallest variant
    }

    print("=" * 60)
    print("Long-Context Benchmark Evaluation")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")

    all_results = {}

    for benchmark_name, task_list in benchmarks.items():
        print(f"\n--- Running {benchmark_name} ---")
        for task in task_list:
            full_task_name = (
                f"{benchmark_name}:{task}" if benchmark_name != "phonebook" else task
            )
            results = run_benchmark(model, full_task_name, output_dir)
            if results:
                all_results[full_task_name] = results

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)

    for task_name, results in all_results.items():
        print(f"\n{task_name}:")
        if "results" in results:
            for subtask, metrics in results["results"].items():
                print(f"  {subtask}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON WITH OFFICIAL IMPLEMENTATIONS")
    print("=" * 60)
    print("""
Note: For full comparison with official implementations, you would need to:

1. **LongBench v2**: Compare with results from https://github.com/THUDM/LongBench
   - Official leaderboard: https://longbench.github.io/

2. **Babilong**: Compare with https://github.com/booydar/babilong
   - Official results in paper: https://arxiv.org/abs/2402.10149

3. **InfiniteBench**: Compare with https://github.com/OpenBMB/InfiniteBench
   - Official leaderboard: https://infinitebench.github.io/

4. **Phonebook (Lost in the Middle)**: Compare with https://github.com/nelson-liu/lost-in-the-middle
   - Original paper: https://arxiv.org/abs/2307.03172

To get accurate comparisons:
- Use the same model (e.g., Llama-2-7B, GPT-3.5-turbo)
- Remove the --limit flag to evaluate on full datasets
- Use appropriate context length settings for each benchmark
""")


if __name__ == "__main__":
    main()
