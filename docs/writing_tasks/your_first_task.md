# Your First Task

This guide walks you through creating your first evaluation task from scratch. By the end, you'll have a working YAML-based task that evaluates a model on a HuggingFace dataset.

A more interactive tutorial is available as a Jupyter notebook [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb).

## Setup

Fork the repo, clone it, and install in development mode:

```bash
# After forking...
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout -b <task-name>
pip install -e ".[dev]"
```

We'll walk through two kinds of tasks: a **generative** task (like [gsm8k](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml)) and a **multiple-choice** task (like [sciq](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/sciq/sciq.yaml)).

## Creating a YAML file

Create a new YAML file in a subfolder of `lm_eval/tasks`:

```bash
mkdir -p lm_eval/tasks/<dataset_name>
touch lm_eval/tasks/<dataset_name>/<my_task>.yaml
```

Or copy the template:

```bash
cp -r templates/new_yaml_task lm_eval/tasks/<dataset_name>
```

## Configuring the dataset

All datasets are managed via the HuggingFace [datasets](https://github.com/huggingface/datasets) API. Check if your dataset is on the [HuggingFace Hub](https://huggingface.co/datasets) — if not, consider uploading it.

```yaml
dataset_path: super_glue       # HF Hub dataset name
dataset_name: boolq            # Dataset configuration (null if none needed)
dataset_kwargs: null            # Extra kwargs for datasets.load_dataset()
```

Specify which splits to use:

```yaml
training_split: train           # For few-shot examples (null if none)
validation_split: validation    # Evaluated if no test_split
test_split: null                # Primary evaluation split
```

Tests run on `test_split` if available, otherwise on `validation_split`.

### Few-shot configuration

Control where few-shot examples come from:

```yaml
# Simple: draw from a specific split
fewshot_split: train

# Advanced: full control
fewshot_config:
  sampler: first_n          # "default" (random) or "first_n"
  split: train              # Override fewshot_split
  samples: [...]            # Hardcoded examples (list of dicts)
  doc_to_text: "..."        # Override doc_to_text for fewshot examples
  doc_to_target: "..."      # Override doc_to_target for fewshot examples
  fewshot_delimiter: "\n\n" # Separator between examples
  target_delimiter: " "     # Separator between question and answer
```

All fields in `fewshot_config` are optional — they inherit from the parent `TaskConfig` if not set.

### Preprocessing with `process_docs`

If your dataset needs cleaning or reformatting, create a `utils.py` in the same directory:

```python
# lm_eval/tasks/<dataset_name>/utils.py
def process_docs(dataset):
    def _process_doc(doc):
        return {
            "query": doc["question"].strip(),
            "choices": doc["options"],
            "gold": int(doc["answer"]),
        }
    return dataset.map(_process_doc)
```

Reference it in your YAML:

```yaml
process_docs: !function utils.process_docs
```

## Writing a prompt template

Define the input, target, and (for multiple-choice) answer choices. The simplest approach is to use **prompt formats**, which auto-generate prompt layouts from your field mappings. For custom layouts, use Jinja2 templates or Python functions.

### Using prompt formats (recommended)

The easiest way to define prompts is with [prompt formats](prompt_formats.md). Just map your dataset columns and pick a format:

```yaml
doc_to_text: question          # Dataset column containing the question
doc_to_target: answer          # Dataset column containing the correct answer
doc_to_choice: choices         # Dataset column containing answer options
formats: mcqa                  # Auto-generates A/B/C/D prompt layout
```

This produces prompts like:

```
Question: What is the capital of France?
A. Berlin
B. Madrid
C. Paris
D. London
Answer:
```

The format handles the `output_type`, delimiters, and scoring automatically. Built-in formats: `mcqa`, `cloze`, `generate`, `cot`.

You can also try different formats at runtime without touching the YAML:

```bash
lm-eval run --tasks my_task@mcqa --model hf --model_args pretrained=gpt2
lm-eval run --tasks my_task@generate --model hf --model_args pretrained=gpt2
```

See [Prompt Formats](prompt_formats.md) for all options including custom instructions, choice labels, and multi-format tasks.

### Basic field references

For simple tasks where a format isn't needed, reference dataset columns directly:

```yaml
doc_to_text: startphrase       # Use the "startphrase" column as input
doc_to_target: label           # Use the "label" column as target
```

### Jinja2 templates

For custom prompt layouts beyond what formats provide, use [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) templates:

```yaml
doc_to_text: "{{passage}}\nQuestion: {{question}}?\nAnswer:"
doc_to_target: "{{answer}}"
```

`{{passage}}` is replaced by the value of `doc["passage"]` for each document.

### Multiple-choice tasks (manual)

If you need full control over multiple-choice formatting (instead of using `formats: mcqa`):

```yaml
doc_to_text: "{{passage}}\nQuestion: {{question}}?\nAnswer:"
doc_to_target: label                     # Index of correct answer (integer)
doc_to_choice: ["no", "yes"]             # The answer options
```

Or reference a dataset column that contains the choices:

```yaml
doc_to_text: "{{support.lstrip()}}\nQuestion: {{question}}\nAnswer:"
doc_to_target: 3                         # Correct answer index
doc_to_choice: "{{[distractor1, distractor2, distractor3, correct_answer]}}"
```

!!! warning
    The harness inserts `target_delimiter` (default: `" "`) between `doc_to_text` and `doc_to_target`. Don't add trailing/leading whitespace in your templates.

### Python functions for prompts

For complex formatting that's easier in Python:

```python
# utils.py
def my_doc_to_text(doc):
    return f"Question: {doc['question'].strip()}\nAnswer:"
```

```yaml
doc_to_text: !function utils.my_doc_to_text
```

## Setting metrics

Choose how to score your task. For most tasks, the defaults work:

- **Multiple-choice tasks**: `acc` (accuracy) and `acc_norm` (length-normalized accuracy) are applied automatically
- **Generation tasks**: `exact_match` is applied automatically

For custom metrics:

```yaml
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
  - metric: bleu
```

`aggregation` and `higher_is_better` can be omitted for built-in metrics — they have sensible defaults. Extra fields (like `ignore_case`) are passed as kwargs to the metric function.

For full details on metrics, scorers, and filter pipelines, see [Scoring & Metrics](scoring_and_metrics.md).

## Registering your task

Give your task a name:

```yaml
task: my_custom_task
```

Optionally, add tags for categorization:

```yaml
tag:
  - multiple_choice
  - science
```

Tags let users run all tasks with a given tag: `lm-eval run --tasks science`.

For tasks outside `lm_eval/tasks/`, use `--include_path`:

```bash
lm-eval run --tasks my_task --include_path /path/to/my/tasks/
```

## Testing your task

### Validate the config

```bash
lm-eval validate --tasks my_custom_task
```

### Inspect prompts

```bash
lm-eval run --model hf --model_args pretrained=gpt2 \
    --tasks my_custom_task --write_out --limit 5
```

!!! tip
    Enable debug logging for detailed output: `export LMEVAL_LOG_LEVEL="DEBUG"`

### Run a quick evaluation

```bash
lm-eval run --model hf --model_args pretrained=gpt2 \
    --tasks my_custom_task --limit 100
```

## Complete example

Here's a complete YAML for a multiple-choice task:

```yaml
task: my_boolq
dataset_path: super_glue
dataset_name: boolq
test_split: validation
doc_to_text: "{{passage}}\nQuestion: {{question}}?\nAnswer:"
doc_to_target: label
doc_to_choice: ["no", "yes"]
metadata:
  version: 1.0
```

And a generative task:

```yaml
task: my_gsm8k
dataset_path: gsm8k
dataset_name: main
test_split: test
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
output_type: generate_until
generation_kwargs:       # Note: YAML field is "generation_kwargs", CLI flag is "--gen_kwargs"
  until: ["\n\n"]
  do_sample: false
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

## Using YAML `include` for templates

Base your YAML on another file to avoid duplication:

```yaml
include: _default_template.yaml
task: mmlu_anatomy
dataset_name: anatomy
```

The included file provides shared fields (dataset_path, doc_to_text, etc.), and your file overrides specific values.

## Versioning

Add a version to your task for reproducibility:

```yaml
metadata:
  version: 1.0
```

Bump the version whenever you make a breaking change. Add a changelog entry to your task's README.

## Submitting your task

Push your work and open a pull request to `main`. See the [Contributing guide](../contributing.md) for code style and testing requirements.

## Next steps

- [Prompt Formats](prompt_formats.md) — declarative prompt layouts without Jinja
- [Scoring & Metrics](scoring_and_metrics.md) — custom metrics, scorers, and filter pipelines
- [Filters](filters.md) — post-processing model outputs before scoring
- [Groups & Benchmarks](groups_and_benchmarks.md) — organizing tasks into groups with aggregate metrics
- [Task Configuration Reference](task_config_reference.md) — complete field reference
