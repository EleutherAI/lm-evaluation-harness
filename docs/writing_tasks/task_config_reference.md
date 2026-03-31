# Task Configuration Reference

Complete reference for all `TaskConfig` fields. For a tutorial introduction, see [Your First Task](your_first_task.md).

## Task naming and registration

| Field | Type | Default | Description |
|---|---|---|---|
| `task` | str | required | Task name. Must be unique. Used to invoke the task from CLI. |
| `task_alias` | str | `null` | Display name for the results table. |
| `tag` | str or list | `null` | Tag(s) for categorization. Enables batch selection via `--tasks tag_name`. |

## Dataset configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `dataset_path` | str | required | HuggingFace Hub dataset name, or a local path. |
| `dataset_name` | str | `null` | Dataset configuration name (the second argument to `datasets.load_dataset()`). |
| `dataset_kwargs` | dict | `null` | Extra kwargs passed to `datasets.load_dataset()` (e.g., `data_files`, `data_dir`). |
| `custom_dataset` | Callable | `null` | Function returning `dict[str, Dataset]`. Receives `metadata` and `model_args` at runtime. |
| `training_split` | str | `null` | Name of the training split. |
| `validation_split` | str | `null` | Name of the validation split. |
| `test_split` | str | `null` | Name of the test split (primary evaluation split). |
| `fewshot_split` | str | `null` | Split to draw few-shot examples from. |
| `process_docs` | Callable | `null` | Function to preprocess each dataset split before prompting. Use `!function utils.my_fn`. |

### Using local datasets

```yaml
# JSON files
dataset_path: json
dataset_kwargs:
  data_files: /path/to/my/data.json

# Pre-split Arrow files
dataset_path: arrow
dataset_kwargs:
  data_files:
    train: /path/to/train/data.arrow
    validation: /path/to/validation/data.arrow

# Previously downloaded HF dataset (via save_to_disk)
dataset_path: hellaswag
dataset_kwargs:
  data_dir: hellaswag_local/
```

You can also set the `LM_EVAL_DATASET_DIR` environment variable as a fallback directory for local datasets.

## Prompting and in-context formatting

| Field | Type | Default | Description |
|---|---|---|---|
| `doc_to_text` | str or Callable | `null` | Jinja2 template, dataset column name, or `!function` producing the input prompt. |
| `doc_to_target` | str or Callable | `null` | Jinja2 template, column name, integer index, or `!function` producing the target. |
| `doc_to_choice` | str or Callable | `null` | Jinja2 template, column name, list, or `!function` producing answer choices (for `multiple_choice`). |
| `description` | str | `null` | Jinja2 template or string prepended before few-shot examples. |
| `use_prompt` | str | `null` | Promptsource template name (e.g., `"promptsource:GPT-3 Style"`). Overrides `doc_to_text`/`doc_to_target`/`doc_to_choice`. |
| `formats` | str or dict | `null` | Prompt format name or config. See [Prompt Formats](prompt_formats.md). |
| `target_delimiter` | str | `" "` | String inserted between `doc_to_text` and `doc_to_target`. |
| `fewshot_delimiter` | str | `"\n\n"` | String inserted between few-shot examples. |
| `gen_prefix` | str | `null` | String appended after the assistant token (or end of prompt without chat templates). |

### Few-shot configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `num_fewshot` | int | `0` | Number of few-shot examples. |
| `fewshot_config` | dict | `null` | Advanced few-shot settings (see below). |

`fewshot_config` fields (all optional, inherit from parent `TaskConfig`):

| Field | Description |
|---|---|
| `sampler` | `"default"` (random) or `"first_n"` |
| `split` | Dataset split for examples (overrides `fewshot_split`) |
| `samples` | Hardcoded list of example dicts |
| `doc_to_text` | Override formatting for few-shot examples |
| `doc_to_target` | Override target formatting for few-shot examples |
| `doc_to_choice` | Override choices for few-shot examples |
| `gen_prefix` | Prefix for assistant responses in few-shot examples |
| `fewshot_delimiter` | Override delimiter between examples |
| `target_delimiter` | Override delimiter between question and answer |

## Scoring

| Field | Type | Default | Description |
|---|---|---|---|
| `output_type` | str | `"generate_until"` | Model request type: `generate_until`, `loglikelihood`, `loglikelihood_rolling`, or `multiple_choice`. |
| `metric_list` | list | `null` | List of `MetricConfig` entries. See [Scoring & Metrics](scoring_and_metrics.md). |
| `filter_list` | list | `null` | List of filter pipelines. See [Filters](filters.md). |
| `scorer` | dict | `null` | Scorer configuration. See [Scoring & Metrics](scoring_and_metrics.md#scorers). |
| `generation_kwargs` | dict | `null` | Generation arguments (e.g., `temperature`, `max_gen_toks`, `until`). Note: the CLI flag is `--gen_kwargs` but the YAML field is `generation_kwargs`. |
| `repeats` | int | `1` | Number of repeated runs per sample. Used for self-consistency or sampling. |

## Other

| Field | Type | Default | Description |
|---|---|---|---|
| `batch_size` | int | `1` | Batch size for this task. |
| `should_decontaminate` | bool | `false` | Whether to perform test set decontamination. |
| `doc_to_decontamination_query` | str | `null` | Query for decontamination (defaults to `doc_to_text` if not set). |
| `metadata` | dict | `null` | Arbitrary metadata. Special keys: `version` (task version), `num_fewshot` (override displayed n-shot). Also passed to `custom_dataset` if defined. |

## YAML features

### `include` — template inheritance

Base your YAML on another file:

```yaml
include: _default_template.yaml
task: mmlu_anatomy
dataset_name: anatomy
```

The included file provides shared fields, and your file overrides specific values. The include path is relative to the including file's directory unless an absolute path is given.

### `!function` — embedded Python

Reference Python functions in your task directory:

```yaml
doc_to_text: !function utils.my_doc_to_text
process_docs: !function utils.process_docs
metric_list:
  - metric: !function utils.my_metric
    aggregation: !function utils.my_aggregation
```

The function script must be in the same directory as the YAML file. Supported for: `doc_to_text`, `doc_to_target`, `doc_to_choice`, `process_docs`, `metric` in `metric_list`, `aggregation` in `metric_list`.

### Python class-based tasks

For tasks that require full Python control:

```yaml
task: squadv2
class: !function task.SQuAD2
```

Custom arguments can be passed to the class:

```yaml
task: my_task
class: !function task.MyTaskClass
recipe: card=cards.my_card,template=templates.my_template
```

## Versioning

```yaml
metadata:
  version: 1.0
```

Bump the version for any breaking change to the task config. Document changes in the task's README:

```
- [Mar 2026] (PR #1234) Version 1.0 -> 2.0: Updated prompt format for consistency.
```

## Good reference tasks

| Type | Example |
|---|---|
| Multiple choice | `lm_eval/tasks/sciq/sciq.yaml` |
| Corpus perplexity | `lm_eval/tasks/wikitext/wikitext.yaml` |
| Generative | `lm_eval/tasks/gsm8k/gsm8k.yaml` |
| Complex filtering | `lm_eval/tasks/gsm8k/gsm8k-cot-self-consistency.yaml` |
| Using formats | Any task with `formats: mcqa` |
