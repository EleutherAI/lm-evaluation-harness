#!/usr/bin/env python3
"""
Parallel Model Evaluation Script

Runs multiple lm_eval commands in parallel across available GPUs.
As each GPU finishes evaluating a model, it automatically picks up the next one from the queue.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm


# ============================================================================
# CONFIGURATION - Edit these values for your use case
# ============================================================================

# List of models to evaluate
MODELS = [
    "mistralai/Mixtral-8x7B-v0.1",
    "moonshotai/Kimi-K2-Base",
    "bertin-project/Gromenauer-7B",
    "indobenchmark/indogpt",
    "lelapa/InkubaLM-0.4B",
    "Qwen/Qwen2.5-0.5B",
    "bigscience/bloom-560m",
    "facebook/xglm-564M",
    "meta-llama/Llama-3.2-1B",
    "sail/Sailor2-1B",
    "Azurro/APT3-1B-Base",
    "CraneAILabs/swahili-gemma-1b",
    "CraneAILabs/ganda-gemma-1b",
    "sapienzanlp/Minerva-1B-base-v1.0",
    "bigscience/bloom-1b1",
    "TucanoBR/Tucano-1b1",
    "kakaocorp/kanana-1.5-2.1b-base",
    "UBC-NLP/cheetah-1.2B",
    "croissantllm/CroissantLLMChat-v0.1",
    "AI-Sweden-Models/gpt-sw3-1.3b",
    "inceptionai/jais-family-1p3b",
    "Qwen/Qwen2.5-1.5B",
    "SeaLLMs/SeaLLMs-v3-1.5B",
    "speakleash/Bielik-1.5B-v3",
    "facebook/xglm-1.7B",
    "bigscience/bloom-1b7",
    "BSC-LT/salamandra-2b",
    "TucanoBR/Tucano-2b4",
    "vilm/vinallama-2.7b",
    "inceptionai/jais-family-2p7b",
    "facebook/xglm-2.9B",
    "bigscience/bloom-3b",
    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B",
    "sapienzanlp/Minerva-3B-base-v1.0",
    "UBC-NLP/cheetah-base",
    "facebook/xglm-4.5B",
    "speakleash/Bielik-4.5B-v3",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2",
    "inceptionai/jais-family-6p7b",
    "universitytehran/PersianMind-v1.0",
    "mistralai/Mistral-7B-v0.1",
    "Qwen/Qwen2.5-7B",
    "SeaLLMs/SeaLLMs-v3-7B",
    "BSC-LT/salamandra-7b",
    "vilm/vinallama-7b",
    "tiiuae/falcon-7b",
    "Unbabel/TowerBase-7B-v0.1",
    "LumiOpen/Viking-7B",
    "Yellow-AI-NLP/komodo-7b-base",
    "ilsp/Meltemi-7B-v1.5",
    "sapienzanlp/Minerva-7B-base-v1.0",
    "bigscience/bloom-7b1",
    "facebook/xglm-7.5B",
    "vinai/PhoGPT-7B5",
    "nvidia/nemotron-3-8b-base-4k",
    "swiss-ai/Apertus-8B-2509",
    "meta-llama/Llama-3.1-8B",
    "aisingapore/Llama-SEA-LION-v3-8B-IT",
    "kakaocorp/kanana-1.5-8b-base",
    "sail/Sailor2-8B",
    "LumiOpen/Llama-Poro-2-8B-base",
    "ilsp/Llama-Krikri-8B-Base",
    "polyglots/SinLlama_v01",
    "utter-project/EuroLLM-9B",
    "aisingapore/Gemma-SEA-LION-v3-9B-IT",
    "Tower-Babel/Babel-9B",
    "Gen2B/HyGPT-10b",
    "Unbabel/TowerBase-13B-v0.1",
    "LumiOpen/Viking-13B",
    "inceptionai/jais-family-13b",
    "Qwen/Qwen2.5-14B",
    "sail/Sailor2-20B",
    "AI-Sweden-Models/gpt-sw3-20b",
    "inceptionai/jais-family-30b-8k",
    "Qwen/Qwen2.5-32B",
    "LumiOpen/Viking-33B",
    "tiiuae/falcon-40b",
    "AI-Sweden-Models/gpt-sw3-40b",
    "swiss-ai/Apertus-70B-2509",
    "meta-llama/Llama-3.1-70B",
    "aisingapore/Llama-SEA-LION-v3-70B-IT",
    "LumiOpen/Llama-Poro-2-70B-base",
    "Qwen/Qwen2.5-72B",
    "Tower-Babel/Babel-83B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "sail/Sailor2-1B-Chat",
    "Azurro/APT3-1B-Instruct-v1",
    "TucanoBR/Tucano-1b1-Instruct",
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "kakaocorp/kanana-1.5-2.1b-instruct-2505",
    "croissantllm/CroissantLLMBase",
    "AI-Sweden-Models/gpt-sw3-1.3b-instruct",
    "inceptionai/jais-family-1p3b-chat",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "SeaLLMs/SeaLLMs-v3-1.5B-Chat",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "speakleash/Bielik-1.5B-v3.0-Instruct",
    "BSC-LT/salamandra-2b-instruct",
    "TucanoBR/Tucano-2b4-Instruct",
    "vilm/vinallama-2.7b-chat",
    "inceptionai/jais-family-2p7b-chat",
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "speakleash/Bielik-4.5B-v3.0-Instruct",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct",
    "inceptionai/jais-family-6p7b-chat",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "SeaLLMs/SeaLLMs-v3-7B-Chat",
    "BSC-LT/salamandra-7b-instruct",
    "vilm/vinallama-7b-chat",
    "tiiuae/falcon-7b-instruct",
    "Unbabel/TowerInstruct-7B-v0.1",
    "ilsp/Meltemi-7B-Instruct-v1.5",
    "sapienzanlp/Minerva-7B-instruct-v1.0",
    "vinai/PhoGPT-7B5-Instruct",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "swiss-ai/Apertus-8B-Instruct-2509",
    "meta-llama/Llama-3.1-8B-Instruct",
    "aisingapore/Llama-SEA-LION-v3-8B",
    "kakaocorp/kanana-1.5-8b-instruct-2505",
    "sail/Sailor2-8B-Chat",
    "LumiOpen/Llama-Poro-2-8B-Instruct",
    "ilsp/Llama-Krikri-8B-Instruct",
    "utter-project/EuroLLM-9B-Instruct",
    "aisingapore/Gemma-SEA-LION-v3-9B",
    "Tower-Babel/Babel-9B-Chat",
]

# Number of GPUs available (will use cuda:0, cuda:1, ..., cuda:N-1)
NUM_GPUS = 8

# Common evaluation parameters
TASKS = "global_piqa"
OUTPUT_PATH = "mrl_test_run"
LIMIT = None  # Set to None to run on full dataset
ADDITIONAL_ARGS = [
    "--log_samples",
]

# Optional: Override batch size, max length, etc.
# ADDITIONAL_ARGS.append("--batch_size=8")


# ============================================================================
# Script Logic - No need to edit below this line
# ============================================================================


def build_eval_command(model: str, gpu_id: int, limit: int | None = None) -> list[str]:
    """Build the lm_eval command for a specific model and GPU."""
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model}",
        "--tasks",
        TASKS,
        "--output_path",
        OUTPUT_PATH,
        "--device",
        f"cuda:{gpu_id}",
        "--batch_size",
        "16",
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    cmd.extend(ADDITIONAL_ARGS)

    return cmd


def run_evaluation(model: str, gpu_id: int) -> dict[str, Any]:
    """Run evaluation for a single model on a specific GPU."""
    cmd = build_eval_command(model, gpu_id, LIMIT)
    cmd_str = " ".join(cmd)

    print(f"[GPU {gpu_id}] Starting: {model}")
    print(f"[GPU {gpu_id}] Command: {cmd_str}\n")

    try:
        # Run the command and wait for it to complete
        # Capture stderr but let stdout go to terminal
        result = subprocess.run(
            cmd, check=True, stdout=sys.stdout, stderr=subprocess.PIPE, text=True
        )

        print(f"\n[GPU {gpu_id}] ✓ Completed: {model}\n")
        return {
            "model": model,
            "gpu_id": gpu_id,
            "status": "success",
            "returncode": result.returncode,
            "command": cmd_str,
            "timestamp": datetime.now().isoformat(),
        }

    except subprocess.CalledProcessError as e:
        print(f"\n[GPU {gpu_id}] ✗ Failed: {model}")
        print(f"[GPU {gpu_id}] Return code: {e.returncode}")
        if e.stderr:
            print(f"[GPU {gpu_id}] Error output: {e.stderr[:500]}\n")  # First 500 chars
        return {
            "model": model,
            "gpu_id": gpu_id,
            "status": "failed",
            "returncode": e.returncode,
            "command": cmd_str,
            "stderr": e.stderr if e.stderr else "",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"\n[GPU {gpu_id}] ✗ Error: {model}")
        print(f"[GPU {gpu_id}] Exception: {e}\n")
        return {
            "model": model,
            "gpu_id": gpu_id,
            "status": "error",
            "error": str(e),
            "command": cmd_str,
            "timestamp": datetime.now().isoformat(),
        }


def has_results(model: str, output_path: str) -> bool:
    """Check if results already exist for this model."""
    # lm_eval typically saves results with the model name sanitized
    # We'll check if any JSON results exist in the output directory
    output_dir = Path(output_path)
    if not output_dir.exists():
        return False

    # Common patterns for result files
    model_name = model.split("/")[-1]  # Get just the model name without org
    patterns = [
        f"*{model_name}*.json",
        f"results_{model_name}*.json",
    ]

    for pattern in patterns:
        if list(output_dir.glob(pattern)):
            return True

    return False


def save_failed_models(results: list[dict[str, str]], output_path: str):
    """Save failed models to both text and JSON files for later debugging."""
    failed_results = [r for r in results if r["status"] != "success"]

    if not failed_results:
        return

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save simple text file with just model names
    failed_txt = output_dir / "failed_models.txt"
    with open(failed_txt, "w") as f:
        for r in failed_results:
            f.write(f"{r['model']}\n")

    # Save detailed JSON file with full error information
    failed_json = output_dir / "failed_models.json"
    with open(failed_json, "w") as f:
        json.dump(failed_results, f, indent=2)

    print("\n📝 Failed models saved to:")
    print(f"   - {failed_txt} (simple list)")
    print(f"   - {failed_json} (detailed errors)")


def load_failed_models(output_path: str) -> list[str]:
    """Load list of previously failed models from text file."""
    failed_txt = Path(output_path) / "failed_models.txt"

    if not failed_txt.exists():
        print(f"No failed models file found at {failed_txt}")
        return []

    with open(failed_txt, "r") as f:
        models = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(models)} failed models from {failed_txt}")
    return models


def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run parallel model evaluations across GPUs"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have results in output directory",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Only run models from the failed_models.txt file",
    )
    args = parser.parse_args()

    # Determine which models to run
    if args.retry_failed:
        models_to_run = load_failed_models(OUTPUT_PATH)
        if not models_to_run:
            print("No failed models to retry. Exiting.")
            return 0
    else:
        models_to_run = MODELS.copy()

    # Filter out models that already have results if --resume is specified
    if args.resume:
        original_count = len(models_to_run)
        models_to_run = [m for m in models_to_run if not has_results(m, OUTPUT_PATH)]
        skipped = original_count - len(models_to_run)
        if skipped > 0:
            print(f"⏭️  Skipping {skipped} models with existing results\n")

    print("=" * 80)
    print("Parallel Model Evaluation")
    print("=" * 80)
    print(f"Models to evaluate: {len(models_to_run)}")
    print(f"GPUs available: {NUM_GPUS}")
    print(f"Tasks: {TASKS}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Limit: {LIMIT if LIMIT else 'Full dataset'}")
    if args.resume:
        print("Mode: Resume (skipping completed models)")
    if args.retry_failed:
        print("Mode: Retry failed models only")
    print("=" * 80)
    print()

    if not models_to_run:
        print("No models to evaluate. Exiting.")
        return 0

    # Create a queue of (model, gpu_id) pairs
    # We cycle through GPUs as we assign models
    model_gpu_pairs = [
        (model, gpu_id % NUM_GPUS) for gpu_id, model in enumerate(models_to_run)
    ]

    results = []
    success_count = 0
    failed_count = 0

    # Use ThreadPoolExecutor to run evaluations in parallel
    # max_workers = NUM_GPUS ensures we don't oversubscribe GPUs
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        # Submit all jobs
        future_to_model = {
            executor.submit(run_evaluation, model, gpu_id): (model, gpu_id)
            for model, gpu_id in model_gpu_pairs
        }

        # Process completed jobs as they finish with a progress bar
        with tqdm(
            total=len(models_to_run), desc="Evaluating models", unit="model"
        ) as pbar:
            for future in as_completed(future_to_model):
                model, gpu_id = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result["status"] == "success":
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Unexpected error processing {model}: {e}")
                    results.append(
                        {
                            "model": model,
                            "gpu_id": gpu_id,
                            "status": "exception",
                            "error": str(e),
                        }
                    )
                    failed_count += 1

                # Update progress bar with current statistics
                pbar.set_postfix({"✓": success_count, "✗": failed_count})
                pbar.update(1)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total models: {len(models_to_run)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed models:")
        for r in results:
            if r["status"] != "success":
                print(f"  - {r['model']} (GPU {r['gpu_id']})")

    print("=" * 80)

    # Save failed models to files for later debugging
    if failed_count > 0:
        save_failed_models(results, OUTPUT_PATH)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
