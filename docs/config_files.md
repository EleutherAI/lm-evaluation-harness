# Configuration Guide

This guide explains how to use YAML configuration files with `lm-eval` to define reusable evaluation settings.

## Overview

Instead of passing many CLI arguments, you can define evaluation parameters in a YAML configuration file:

```bash
# Instead of:
lm-eval run --model hf --model_args pretrained=gpt2,dtype=float32 --tasks hellaswag arc_easy --num_fewshot 5 --batch_size 8 --device cuda:0

# Use:
lm-eval run --config eval_config.yaml
```

CLI arguments override config file values, so you can set defaults in a config file and override specific settings:

```bash
lm-eval run --config eval_config.yaml --tasks mmlu --limit 100
```

## Quick Reference

All configuration keys correspond directly to CLI arguments. See the [CLI Reference](interface.md#lm-eval-run) for detailed descriptions of each option.

## Config Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"hf"` | Model type/provider |
| `model_args` | dict | `{}` | Model constructor arguments |
| `tasks` | list/string | required | Tasks to evaluate |
| `num_fewshot` | int/null | `null` | Few-shot example count |
| `batch_size` | int/string | `1` | Batch size or "auto" |
| `max_batch_size` | int/null | `null` | Max batch size for auto |
| `device` | string/null | `"cuda:0"` | Device to use |
| `limit` | float/null | `null` | Example limit per task |
| `samples` | dict/null | `null` | Specific sample indices |
| `use_cache` | string/null | `null` | Response cache path |
| `cache_requests` | string/dict | `{}` | Request cache settings |
| `output_path` | string/null | `null` | Results output path |
| `log_samples` | bool | `false` | Save model I/O |
| `predict_only` | bool | `false` | Skip metrics |
| `apply_chat_template` | bool/string | `false` | Chat template |
| `system_instruction` | string/null | `null` | System prompt |
| `fewshot_as_multiturn` | bool/null | `null` | Multi-turn few-shot |
| `include_path` | string/null | `null` | External tasks path |
| `gen_kwargs` | dict | `{}` | Generation arguments |
| `wandb_args` | dict | `{}` | W&B init arguments |
| `hf_hub_log_args` | dict | `{}` | HF Hub logging |
| `seed` | list/int | `[0,1234,1234,1234]` | Random seeds |
| `trust_remote_code` | bool | `false` | Trust remote code |
| `metadata` | dict | `{}` | Task metadata |

---

## Example

```yaml
# basic_eval.yaml
model: hf
model_args:
  pretrained: gpt2
  dtype: float32

tasks:
  - hellaswag
  - arc_easy

num_fewshot: 0
batch_size: auto
device: cuda:0

output_path: ./results/gpt2/
log_samples: true

wandb_args:
  project: llm-evals
  name: mistral-7b-instruct
  tags:
    - mistral
    - instruct
    - production

hf_hub_log_args:
  hub_results_org: my-org
  results_repo_name: llm-eval-results
  push_results_to_hub: true
  public_repo: false
```

---

## Programmatic Usage

For loading config files in Python, see the [Python API Guide](python-api.md#using-evaluatorconfig).

---

## Validation

Validate your configuration before running:

```bash
# Check that tasks exist
lm-eval validate --tasks hellaswag,arc_easy

# With external tasks
lm-eval validate --tasks my_task --include_path /path/to/tasks
```

---

## Tips

1. **Start simple**: Begin with minimal config and add options as needed
2. **Use CLI overrides**: Set defaults in config, override with CLI for experiments
3. **Separate concerns**: Create different configs for different model families or task sets
4. **Version control**: Commit config files alongside results for reproducibility
5. **Use comments**: YAML supports `#` comments to document your choices

## Self-consency (sampling-based) evaluation

For sampling-based evaluation with `repeats` > 1, set `do_sample: true` in
`generation_kwargs` and combine the `repeats` config field with a filter
pipeline so the sampled responses are actually aggregated. With the default
`take_first` filter, all but the first of the sampled responses are
discarded (a warning is emitted in this case).

Majority voting (chain an extraction filter with `majority_vote`):

```yaml
task: my_task_sc
output_type: generate_until
generation_kwargs:
  do_sample: true
  temperature: 0.7
repeats: 8
filter_list:
  - name: "sc"
    filter:
      - function: "regex"
        regex_pattern: "#### (-?[0-9.,]+)"
      - function: "majority_vote"
metric_list:
  - metric: exact_match
```

Pass@k (retain all k responses with `take_first_k`, then score any-correct):

```yaml
filter_list:
  - name: "sc"
    filter:
      - function: "regex"
        regex_pattern: "#### (-?[0-9.,]+)"
      - function: "take_first_k"
        k: 8
metric_list:
  - metric: pass_at_k
```

Both `repeats` and the filter pipeline can also be set from the Python API:
`simple_evaluate(..., repeats=8, gen_kwargs="do_sample=True,temperature=0.7")`
together with a task config that defines the `filter_list` above.
