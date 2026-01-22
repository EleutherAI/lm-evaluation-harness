# User Guide

This document details the interface exposed by `lm-eval` and provides details on what flags are available to users.

## Command-line Interface

The `lm-eval` CLI is organized into subcommands:

| Command | Description |
|---------|-------------|
| `lm-eval run` | Run evaluations on language models |
| `lm-eval ls` | List available tasks, groups, subtasks, or tags |
| `lm-eval validate` | Validate task configurations |

Run the library via the `lm-eval` entrypoint or `python -m lm_eval`.

Use `-h` or `--help` to see available options:

```bash
lm-eval -h              # Show all subcommands
lm-eval run -h          # Show options for run command
lm-eval ls -h           # Show options for list command
```

> **Legacy Compatibility**: The original single-command interface still works. Running `lm-eval --model hf --tasks hellaswag` automatically inserts the `run` subcommand.

---

## Quick Start

```bash
# List available tasks
lm-eval ls tasks

# Basic evaluation
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag

# With few-shot examples
lm-eval run --model hf --model_args pretrained=gpt2 --tasks arc_easy --num_fewshot 5

# Save results and model outputs
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --output_path ./results/ --log_samples

# Use a config file
lm-eval run --config eval_config.yaml
```

---

## `lm-eval run`

Run evaluations on language models.

```bash
lm-eval run --model <model> --tasks <task> [options]
```

### Quick Examples

```bash
# Basic evaluation with HuggingFace model
lm-eval run --model hf --model_args pretrained=gpt2 dtype=float32 --tasks hellaswag

# Multiple tasks with few-shot examples
lm-eval run --model vllm --model_args pretrained=EleutherAI/gpt-j-6B --tasks arc_easy arc_challenge --num_fewshot 5

# Custom generation parameters
lm-eval run --model hf --model_args pretrained=gpt2 --tasks lambada --gen_kwargs temperature=0.8 top_p=0.95

# Use a YAML configuration file
lm-eval run --config my_config.yaml --tasks mmlu
```

### Model and Tasks

| Argument | Short | Description |
|----------|-------|-------------|
| `--model` | `-M` | Model type/provider name (default: `hf`). See [supported models](https://github.com/EleutherAI/lm-evaluation-harness#model-apis-and-inference-servers). |
| `--model_args` | `-a` | Model constructor arguments as `key=val key2=val2` or `key=val,key2=val2`. For HuggingFace models, see [`HFLM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py) for available arguments. |
| `--tasks` | `-t` | Space or comma-separated list of task names or groups. Use `lm-eval ls tasks` to see available tasks. |
| `--apply_chat_template` | | Apply chat template to prompts. Use without argument for default template, or specify template name. |
| `--limit` | `-L` | Limit examples per task. Integer for count, float (0.0-1.0) for percentage. **For testing only.** |
| `--use_cache` | `-c` | Path prefix for SQLite cache of model responses (e.g., `/path/to/cache_`). |

### Evaluation Settings

| Argument | Short | Description |
|----------|-------|-------------|
| `--num_fewshot` | `-f` | Number of few-shot examples in context. |
| `--batch_size` | `-b` | Batch size: integer, `auto`, or `auto:N` to auto-tune N times (default: 1). |
| `--max_batch_size` | | Maximum batch size when using `--batch_size auto`. |
| `--device` | | Device to use: `cuda`, `cuda:0`, `cpu`, `mps` (default: `cuda`). |
| `--gen_kwargs` | | Generation arguments as `key=val key2=val2`. Values parsed with `ast.literal_eval`. Example: `temperature=0.8 'stop=["\n\n"]'` |

### Data and Output

| Argument | Short | Description |
|----------|-------|-------------|
| `--output_path` | `-o` | Output directory or JSON file for results. Required with `--log_samples`. |
| `--log_samples` | `-s` | Save all model inputs/outputs for post-hoc analysis. |
| `--samples` | `-E` | JSON mapping task names to sample indices, e.g., `'{"task1": [0,1,2]}'`. Incompatible with `--limit`. |

### Caching and Performance

| Argument | Description |
|----------|-------------|
| `--cache_requests` | Cache preprocessed prompts: `true`, `refresh`, or `delete`. Cached files stored in `lm_eval/cache/.cache` or path set by `LM_HARNESS_CACHE_PATH` env var. |
| `--check_integrity` | Run task test suite validation before evaluation. |

### Prompt Formatting

| Argument | Description |
|----------|-------------|
| `--system_instruction` | Custom system instruction prepended to prompts. |
| `--fewshot_as_multiturn` | Format few-shot examples as multi-turn conversation. Auto-enabled with `--apply_chat_template`. Set to `false` to disable. |

### Task Management

| Argument | Description |
|----------|-------------|
| `--include_path` | Additional directory containing external task YAML files. |

### Logging and Tracking

| Argument | Short | Description |
|----------|-------|-------------|
| `--verbosity` | `-v` | **(Deprecated)** Use `LMEVAL_LOG_LEVEL` env var instead. |
| `--write_out` | `-w` | Print prompts for first few documents (for debugging). |
| `--show_config` | | Display full task configuration after evaluation. |
| `--wandb_args` | | Weights & Biases arguments as `key=val`. E.g., `project=my-project name=run-1`. |
| `--wandb_config_args` | | Additional W&B config arguments. |
| `--hf_hub_log_args` | | HuggingFace Hub logging arguments. See [HF Hub Logging](#huggingface-hub-logging). |

### Advanced Options

| Argument | Short | Description |
|----------|-------|-------------|
| `--predict_only` | `-x` | Save predictions only, skip metric computation. Implies `--log_samples`. |
| `--seed` | | Random seeds as single integer or comma-separated list for `python,numpy,torch,fewshot`. Default: `0,1234,1234,1234`. Use `None` to skip. Example: `--seed 42` or `--seed 0,None,8,52`. |
| `--trust_remote_code` | | Allow executing remote code from HuggingFace Hub. |
| `--confirm_run_unsafe_code` | | Confirm understanding of risks for tasks executing arbitrary Python. |
| `--metadata` | | JSON string passed to TaskConfig. Required for some tasks like RULER. Example: `--metadata '{"max_seq_length": 4096}'`. |

### Configuration File

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | `-C` | Path to YAML configuration file. CLI arguments override config file values. See [Configuration Files](config_files.md). |

### HuggingFace Hub Logging

The `--hf_hub_log_args` argument accepts these keys:

| Key | Description |
|-----|-------------|
| `hub_results_org` | Organization name on HF Hub. Defaults to token owner. |
| `details_repo_name` | Repository name for detailed results. |
| `results_repo_name` | Repository name for aggregated results. |
| `push_results_to_hub` | `True`/`False` - push results to Hub. |
| `push_samples_to_hub` | `True`/`False` - push samples to Hub. Requires `--log_samples`. |
| `public_repo` | `True`/`False` - make repository public. |
| `leaderboard_url` | URL to associated leaderboard. |
| `point_of_contact` | Contact email for results dataset. |
| `gated` | `True`/`False` - gate the details dataset. |

---

## `lm-eval ls`

List available tasks, groups, subtasks, or tags.

```bash
lm-eval ls [tasks|groups|subtasks|tags] [--include_path DIR]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `tasks` | List all available tasks (groups, subtasks, and tags). |
| `groups` | List only task groups (e.g., `mmlu`, `glue`, `superglue`). |
| `subtasks` | List only individual subtasks (e.g., `mmlu_anatomy`, `hellaswag`). |
| `tags` | List task tags (e.g., `reasoning`, `knowledge`). |
| `--include_path` | Additional directory for external task definitions. |

### Task Organization

- **Groups**: Collections of related tasks with aggregated metrics across subtasks (e.g., `mmlu` contains 57 subtasks)
- **Subtasks**: Individual evaluation tasks (e.g., `mmlu_anatomy`, `hellaswag`)
- **Tags**: Categories for filtering tasks without aggregated metrics (e.g., `reasoning`, `language`)

### Examples

```bash
# List all tasks
lm-eval ls tasks

# List only task groups
lm-eval ls groups

# Include external tasks
lm-eval ls tasks --include_path /path/to/external/tasks
```

---

## `lm-eval validate`

Validate task configurations before running evaluations.

```bash
lm-eval validate --tasks <task1,task2> [--include_path DIR]
```

### Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--tasks` | `-t` | **(Required)** Comma-separated list of task names to validate. |
| `--include_path` | | Additional directory for external task definitions. |

### Validation Checks

The validate command performs:

- **Task existence**: Verifies all specified tasks are available
- **Configuration syntax**: Checks YAML/JSON configuration files
- **Dataset access**: Validates dataset paths and configurations
- **Required fields**: Ensures all mandatory task parameters are present
- **Metric definitions**: Verifies metric functions and aggregation methods
- **Filter pipelines**: Validates filter chains and their parameters
- **Template rendering**: Tests prompt templates with sample data

### Examples

```bash
# Validate a single task
lm-eval validate --tasks hellaswag

# Validate multiple tasks
lm-eval validate --tasks arc_easy,arc_challenge,hellaswag

# Validate a task group
lm-eval validate --tasks mmlu

# Validate external tasks
lm-eval validate --tasks my_custom_task --include_path ./custom_tasks
```

---

## Python API

For programmatic usage, see the [Python API Guide](python-api.md).

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LMEVAL_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LM_HARNESS_CACHE_PATH` | Path for cached requests (default: `lm_eval/cache/.cache`). |
| `HF_TOKEN` | HuggingFace Hub token for private datasets/models. |
| `TOKENIZERS_PARALLELISM` | Set to `false` to avoid tokenizer warnings (auto-set by CLI). |
