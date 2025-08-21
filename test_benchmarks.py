#!/usr/bin/env python3
"""
Test script for long-context benchmarks - runs on CPU with small samples
"""
import os
import json
import subprocess
from datetime import datetime

def test_benchmark(task_name):
    """Test a single benchmark task"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"test_results/{task_name}_{timestamp}"
    
    # Using CPU device and smaller model
    cmd = [
        "python3", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", "pretrained=openai-community/gpt2,device_map=cpu",
        "--tasks", task_name,
        "--batch_size", "1",
        "--output_path", output_path,
        "--limit", "1",  # Just 1 sample for testing
        "--verbosity", "INFO",
        "--log_samples"
    ]
    
    print(f"\nTesting task: {task_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✓ {task_name} ran successfully")
            
            # Try to read results
            results_file = f"{output_path}/results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                    # Extract key metrics
                    if "results" in results:
                        for subtask, metrics in results["results"].items():
                            print(f"  Results for {subtask}:")
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    print(f"    {metric}: {value:.4f}")
                    
                    # Show configs
                    if "configs" in results:
                        for subtask, config in results["configs"].items():
                            if "metadata" in config:
                                meta = config["metadata"]
                                if "version" in meta:
                                    print(f"  Version: {meta['version']}")
                                    
            return True
        else:
            print(f"✗ {task_name} failed")
            if "not a registered task" in result.stderr:
                print("  Error: Task not found. Available tasks might have different names.")
            else:
                print(f"  Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ {task_name} timed out")
        return False
    except Exception as e:
        print(f"✗ {task_name} error: {e}")
        return False

def main():
    # Create test output directory
    os.makedirs("test_results", exist_ok=True)
    
    print("=" * 60)
    print("Testing Long-Context Benchmarks")
    print("=" * 60)
    print("Note: Using GPT-2 on CPU with 1 sample per task for testing")
    
    # Test tasks - using actual task names from the yaml files
    test_tasks = [
        # LongBench v2 tasks
        "longbench_v2_2wikimqa",
        "longbench_v2_book_qa_eng",
        
        # Babilong tasks  
        "babilong_qa1_single_fact",
        
        # InfiniteBench tasks
        "infinitebench_kv_retrieval",
        
        # Phonebook tasks
        "phonebook_1k",
    ]
    
    successful = []
    failed = []
    
    for task in test_tasks:
        if test_benchmark(task):
            successful.append(task)
        else:
            failed.append(task)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(successful)}/{len(test_tasks)}")
    if successful:
        print("  ✓ " + "\n  ✓ ".join(successful))
    if failed:
        print(f"\nFailed: {len(failed)}/{len(test_tasks)}")
        print("  ✗ " + "\n  ✗ ".join(failed))
    
    print("\n" + "=" * 60)
    print("OFFICIAL IMPLEMENTATION COMPARISON NOTES")
    print("=" * 60)
    print("""
To properly compare with official implementations:

1. **LongBench v2** (THUDM)
   - Official repo: https://github.com/THUDM/LongBench
   - Run: Use same models (e.g., Llama-2-7B-chat)
   - Metrics: F1 scores for QA, ROUGE for summarization
   
2. **Babilong** (Booydar)
   - Official repo: https://github.com/booydar/babilong
   - Paper: https://arxiv.org/abs/2402.10149
   - Key: Test on different context lengths (1k, 10k, 100k, 1M+)
   
3. **InfiniteBench** (OpenBMB)
   - Official repo: https://github.com/OpenBMB/InfiniteBench
   - Leaderboard: https://infinitebench.github.io/
   - Focus: 100k+ token tasks
   
4. **Phonebook/Lost in the Middle** (Nelson Liu)
   - Official repo: https://github.com/nelson-liu/lost-in-the-middle
   - Key metric: Accuracy vs. position in context

For accurate comparison:
- Use the same models and prompts
- Run on full datasets (remove --limit flag)
- Match context window sizes
- Use appropriate GPU resources for long contexts
""")

if __name__ == "__main__":
    main()