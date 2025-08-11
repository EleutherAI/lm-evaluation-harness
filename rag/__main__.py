#!/usr/bin/env python3
"""
Interactive RAG Evaluation CLI

A simplified, interactive command-line interface for RAG evaluation tasks.
Usage: python -m rag
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Union, Dict, List

# Available tasks with descriptions
TASKS = {
    "1": ("rag_evaluation", "Comprehensive Evaluation - Evaluate all RAG metrics"),
    "2": ("rag_evaluation_relevance", "Query Relevance - Evaluate response relevance to query"),
    "3": ("rag_evaluation_completeness", "Query Completeness - Evaluate response completeness in answering query"),
    "4": ("rag_evaluation_adherence", "Context Adherence - Evaluate response adherence to context"),
    "5": ("rag_evaluation_context_completeness", "Context Completeness - Evaluate completeness of context usage"),
    "6": ("rag_evaluation_coherence", "Response Coherence - Evaluate response grammar and readability"),
    "7": ("rag_evaluation_length", "Response Length - Evaluate if response length is appropriate"),
    "8": ("rag_evaluation_refusal_quality", "Refusal Quality - Evaluate quality of refusal responses"),
    "9": ("rag_evaluation_refusal_clarification", "Refusal Clarification Quality - Evaluate quality of clarification questions"),
    "10": ("rag_evaluation_refusal_presence", "Refusal Presence - Detect presence of refusal responses"),
    "11": ("rag_evaluation_correctness", "Correctness - Evaluate response correctness")
}

# Available models with descriptions
MODELS = {
    "1": ("hf", "Hugging Face Model"),
    "2": ("vllm", "vLLM Model"),
    "3": ("anthropic", "Anthropic Model"),
    "4": ("openai", "OpenAI Model"),
    "5": ("custom", "Custom Model")
}

def print_banner():
    """Print the RAG CLI banner."""
    print("=" * 60)
    print("RAG Evaluation CLI - Interactive Evaluation Tool")
    print("=" * 60)
    print("Welcome to the simplified RAG evaluation tool!")
    print("Please follow the prompts to make your selections...")
    print()

def get_user_choice(options: Dict[str, tuple], prompt: str) -> str:
    """Get user choice from a list of options."""
    print(f"\n{prompt}")
    print("-" * 40)
    
    for key, (value, description) in options.items():
        print(f"{key}. {description}")
    
    while True:
        choice = input(f"\nPlease select (1-{len(options)}): ").strip()
        if choice in options:
            return options[choice][0]
        else:
            print(f"❌ Invalid selection, please enter a number between 1-{len(options)}")

def validate_hf_model(model_name: str) -> bool:
    """Validate if Hugging Face model exists without loading it."""
    try:
        from huggingface_hub import model_info
        
        print(f"🔍 Validating model: {model_name}")
        
        # Check if model exists on Hugging Face Hub
        info = model_info(model_name)
        
        if info:
            print(f"✅ Model found: {model_name}")
            return True
        else:
            print(f"❌ Model not found: {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        return False

def get_model_args(model_type: str) -> str:
    """Get model arguments based on model type."""
    if model_type == "hf":
        print("\n🤗 Hugging Face Model Configuration")
        print("-" * 30)
        
        while True:
            pretrained = input("Enter model name (e.g., Qwen/Qwen2.5-1.5B-Instruct): ").strip()
            if not pretrained:
                pretrained = "Qwen/Qwen2.5-1.5B-Instruct"
            
            # Validate model exists
            if validate_hf_model(pretrained):
                return f"pretrained={pretrained}"
            else:
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("❌ Model configuration failed. Exiting...")
                    sys.exit(1)
    
    elif model_type == "vllm":
        print("\n📝 vLLM Model Configuration")
        print("-" * 30)
        model = input("Enter model name: ").strip()
        if not model:
            model = "Qwen/Qwen2.5-1.5B-Instruct"
        
        return f"model={model}"
    
    elif model_type == "anthropic":
        print("\n📝 Anthropic Model Configuration")
        print("-" * 30)
        model = input("Enter model name (default: claude-3-sonnet-20240229): ").strip()
        if not model:
            model = "claude-3-sonnet-20240229"
        
        return f"model={model}"
    
    elif model_type == "openai":
        print("\n📝 OpenAI Model Configuration")
        print("-" * 30)
        model = input("Enter model name (default: gpt-4o): ").strip()
        if not model:
            model = "gpt-4o"
        
        return f"model={model}"
    
    else:  # custom
        print("\n📝 Custom Model Configuration")
        print("-" * 30)
        args = input("Enter model parameters (format: key1=value1,key2=value2): ").strip()
        return args

def validate_dataset_structure(dataset, dataset_name: str = "dataset") -> bool:
    """Validate dataset structure for RAG evaluation (RED6k-toy format)."""
    try:
        # Check if dataset has at least one example
        if len(dataset) == 0:
            print(f"❌ {dataset_name} is empty")
            return False
        
        # Get first example to check structure
        first_example = dataset[0]
        
        # Check required fields according to RED6k-toy format
        required_fields = {
            "question": "The user query (string)",
            "contexts": "List of context chunks/documents (list[str])", 
            "answer": "The ground truth response (string)"
        }
        
        missing_fields = []
        found_fields = []
        
        for field, description in required_fields.items():
            if field not in first_example:
                missing_fields.append(f"{field} ({description})")
            else:
                found_fields.append(field)
                
                # Additional validation for specific fields
                if field == "question":
                    if not isinstance(first_example[field], str):
                        print(f"❌ Field '{field}' must be a string")
                        return False
                elif field == "contexts":
                    if not isinstance(first_example[field], list):
                        print(f"❌ Field '{field}' must be a list")
                        return False
                    # Check if all items in contexts are strings
                    for i, ctx in enumerate(first_example[field]):
                        if not isinstance(ctx, str):
                            print(f"❌ Field '{field}' item {i} must be a string")
                            return False
                elif field == "answer":
                    if not isinstance(first_example[field], str):
                        print(f"❌ Field '{field}' must be a string")
                        return False
        
        if missing_fields:
            print(f"❌ {dataset_name} missing required fields:")
            for field in missing_fields:
                print(f"   - {field}")
            return False
        
        # Check for optional model_response field
        if "model_response" in first_example:
            found_fields.append("model_response")
            print("✅ Found 'model_response' field (will be used for evaluation)")
        else:
            print("⚠️  No 'model_response' field found (will be generated during evaluation)")
        
        print(f"✅ {dataset_name} validation successful!")
        print(f"✅ Found required fields: {', '.join(found_fields)}")
        print(f"✅ {dataset_name} size: {len(dataset)} examples")
        
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} validation failed: {str(e)}")
        return False

def validate_local_file(file_path: str) -> bool:
    """Validate local file structure for RAG evaluation."""
    try:
        import json
        import datasets
        
        print(f"🔍 Validating local file: {file_path}")
        
        # Determine file type and load data
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # List of examples
                dataset = data
            elif isinstance(data, dict) and 'data' in data:
                # Dict with 'data' key containing examples
                dataset = data['data']
            elif isinstance(data, dict) and 'examples' in data:
                # Dict with 'examples' key containing examples
                dataset = data['examples']
            else:
                print("❌ JSON file must contain a list of examples or dict with 'data'/'examples' key")
                return False
                
        elif file_path.endswith('.jsonl'):
            # JSONL format - one JSON object per line
            dataset = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            example = json.loads(line)
                            dataset.append(example)
                        except json.JSONDecodeError as e:
                            print(f"❌ Invalid JSON on line {line_num}: {e}")
                            return False
        else:
            print("❌ Unsupported file format. Please use .json or .jsonl files")
            return False
        
        return validate_dataset_structure(dataset, "local file")
        
    except Exception as e:
        print(f"❌ Local file validation failed: {str(e)}")
        return False

def validate_hf_dataset(dataset_path: str) -> bool:
    """Validate Hugging Face dataset structure for RAG evaluation."""
    try:
        import datasets
        from datasets import load_dataset
        
        print(f"🔍 Validating Hugging Face dataset: {dataset_path}")
        
        # Load dataset
        dataset = load_dataset(dataset_path)
        
        # Check if dataset has train split
        if "train" not in dataset:
            print("❌ Dataset must have a 'train' split")
            return False
        
        train_dataset = dataset["train"]
        
        return validate_dataset_structure(train_dataset, "Hugging Face dataset")
        
    except Exception as e:
        print(f"❌ Hugging Face dataset validation failed: {str(e)}")
        return False

def get_data_path() -> str:
    """Get data path from user."""
    print("\n📊 Data Configuration")
    print("-" * 20)
    print("1. Use default example data")
    print("2. Specify custom data file")
    print("3. Use Hugging Face dataset")
    
    choice = input("\nPlease select (1-3): ").strip()
    
    if choice == "1":
        return ""
    elif choice == "2":
        while True:
            path = input("Enter data file path: ").strip()
            if not path:
                print("❌ File path cannot be empty")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("❌ Data configuration failed. Exiting...")
                    sys.exit(1)
                continue
            
            if not os.path.exists(path):
                print(f"❌ File does not exist: {path}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("❌ Data configuration failed. Exiting...")
                    sys.exit(1)
                continue
            
            # Validate file structure
            if validate_local_file(path):
                return path
            else:
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("❌ Data configuration failed. Exiting...")
                    sys.exit(1)
    elif choice == "3":
        while True:
            dataset_path = input("Enter Hugging Face dataset path (e.g., username/dataset_name): ").strip()
            if not dataset_path:
                print("❌ Dataset path cannot be empty")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("❌ Data configuration failed. Exiting...")
                    sys.exit(1)
                continue
            
            if validate_hf_dataset(dataset_path):
                return f"hf://{dataset_path}"
            else:
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("❌ Data configuration failed. Exiting...")
                    sys.exit(1)
    else:
        print("Using default data")
        return ""

def get_output_path() -> str:
    """Get output path from user."""
    print("\n💾 Output Configuration")
    print("-" * 20)
    
    default_path = "result"
    path = input(f"Enter output directory (default: {default_path}): ").strip()
    
    if not path:
        path = default_path
    
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)
    
    return path

def get_device() -> str:
    """Get device from user."""
    print("\n🖥️  Device Configuration")
    print("-" * 20)
    print("1. CPU")
    print("2. CUDA (GPU)")
    print("3. MPS (Apple Silicon GPU)")
    print("4. Auto-detect")
    
    choice = input("\nPlease select (1-4): ").strip()
    
    if choice == "1":
        return "cpu"
    elif choice == "2":
        device_id = input("Enter GPU ID (default: 0): ").strip()
        return f"cuda:{device_id}" if device_id else "cuda:0"
    elif choice == "3":
        return "mps"
    else:
        return None

def get_limit() -> Union[float, None]:
    """Get limit from user."""
    print("\n🔢 Limit Configuration")
    print("-" * 20)
    print("1. No limit (evaluate all data)")
    print("2. Limit number of examples (for testing)")
    
    choice = input("\nPlease select (1-2): ").strip()
    
    if choice == "1":
        return None
    elif choice == "2":
        while True:
            try:
                limit = input("Enter limit number (e.g., 10): ").strip()
                if not limit:
                    return 10
                return float(limit)
            except ValueError:
                print("❌ Please enter a valid number")

def check_openai_api_key() -> bool:
    """Check if OpenAI API key is available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  OpenAI API Key Configuration")
        print("-" * 30)
        print("OPENAI_API_KEY environment variable not set")
        
        choice = input("Set it now? (y/n): ").strip().lower()
        if choice == 'y':
            api_key = input("Enter OpenAI API key: ").strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print("✅ API key set")
                return True
            else:
                print("❌ API key cannot be empty")
                return False
        else:
            return False
    return True

def confirm_configuration(config: Dict) -> bool:
    """Show configuration and ask for confirmation."""
    print("\n" + "=" * 60)
    print("Configuration Confirmation")
    print("=" * 60)
    
    print(f"Task: {config['task']}")
    print(f"Model: {config['model']}")
    print(f"Model Args: {config['model_args']}")
    
    # Display data path appropriately
    if config['data_path']:
        if config['data_path'].startswith('hf://'):
            print(f"Data Path: Hugging Face dataset ({config['data_path'][4:]})")
        else:
            print(f"Data Path: {config['data_path']}")
    else:
        print(f"Data Path: Default data")
    
    print(f"Output Directory: {config['output_path']}")
    print(f"Device: {config['device'] or 'Auto-detect'}")
    print(f"Limit: {config['limit'] or 'No limit'}")
    print(f"Log Samples: Yes (auto-enabled)")
    
    print("\n" + "-" * 60)
    choice = input("Confirm to start evaluation? (y/n): ").strip().lower()
    return choice == 'y'

def run_evaluation(config: Dict):
    """Run the evaluation by directly calling lm_eval evaluator."""
    print("\n🚀 Starting evaluation...")
    print("=" * 60)
    
    # Import lm_eval modules
    from lm_eval import evaluator, utils
    from lm_eval.evaluator import request_caching_arg_to_dict
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import (
        handle_non_serializable,
        make_table,
        simple_parse_args_string,
    )
    
    # Setup logging
    utils.setup_logging("INFO")
    eval_logger = logging.getLogger(__name__)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Setup evaluation tracker
    evaluation_tracker_args = simple_parse_args_string(f"output_path={config['output_path']}")
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)
    
    # Parse model args
    metadata = simple_parse_args_string(config['model_args']) if config['model_args'] else {}
    
    # Setup task manager
    task_manager = TaskManager(metadata=metadata)
    
    # Get task names
    task_names = task_manager.match_tasks([config['task']])
    
    if not task_names:
        eval_logger.error(f"Task '{config['task']}' not found")
        sys.exit(1)
    
    # Update dataset_path if custom dataset is provided
    if config['data_path'] and config['data_path'].startswith('hf://'):
        dataset_path = config['data_path'][4:]  # Remove 'hf://' prefix
        for task_config in task_names:
            if hasattr(task_config, 'dataset_path'):
                task_config.dataset_path = dataset_path
            elif isinstance(task_config, dict):
                task_config['dataset_path'] = dataset_path
    
    eval_logger.info(f"Selected Tasks: {task_names}")
    
    # Setup request caching
    request_caching_args = request_caching_arg_to_dict(cache_requests=None)
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=config['model'],
        model_args=config['model_args'],
        tasks=task_names,
        num_fewshot=None,
        batch_size=1,
        max_batch_size=None,
        device=config['device'],
        use_cache=None,
        limit=config['limit'],
        samples=None,
        check_integrity=False,
        write_out=False,
        log_samples=True,
        evaluation_tracker=evaluation_tracker,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        gen_kwargs=None,
        task_manager=task_manager,
        predict_only=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        confirm_run_unsafe_code=False,
        metadata=metadata,
        **request_caching_args,
    )
    
    if results is not None:
        samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        
        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        
        # Save results
        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples
        )
        
        for task_name, task_config in results["configs"].items():
            evaluation_tracker.save_results_samples(
                task_name=task_name, samples=samples[task_name]
            )
        
        print(
            f"{config['model']} ({config['model_args']}), limit: {config['limit']}, "
            f"batch_size: 1 ({batch_sizes})"
        )
        print(make_table(results))
        
        print(f"\n✅ Evaluation completed! Results saved to directory: {config['output_path']}")

def interactive_cli():
    """Main interactive CLI function."""
    print_banner()
    
    # Check OpenAI API key for RAG tasks
    if not check_openai_api_key():
        print("❌ OpenAI API key required for RAG evaluation")
        sys.exit(1)
    
    # Get task
    task = get_user_choice(TASKS, "🎯 Please select evaluation task:")
    
    # Get model
    model = get_user_choice(MODELS, "🤖 Please select model type:")
    
    # Get model arguments
    model_args = get_model_args(model)
    
    # Get data path
    data_path = get_data_path()
    
    # Get output path
    output_path = get_output_path()
    
    # Get device
    device = get_device()
    
    # Get limit
    limit = get_limit()
    
    # Build configuration
    config = {
        "task": task,
        "model": model,
        "model_args": model_args,
        "data_path": data_path,
        "output_path": output_path,
        "device": device,
        "limit": limit,
    }
    
    # Confirm and run
    if confirm_configuration(config):
        run_evaluation(config)
    else:
        print("\n❌ Evaluation cancelled")
        sys.exit(0)

def cli_rag_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    """Main CLI function - now interactive by default."""
    if args:
        # If args are provided, use the old non-interactive mode
        # This maintains backward compatibility
        print("⚠️  Command line arguments detected, using non-interactive mode")
        # ... (old implementation would go here)
        pass
    else:
        # Interactive mode
        interactive_cli()

if __name__ == "__main__":
    cli_rag_evaluate()