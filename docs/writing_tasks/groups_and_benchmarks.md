# Groups & Benchmarks

Groups let you organize related tasks into named collections with aggregate metrics — essential for benchmarks like MMLU (57 subtasks) or SuperGLUE.

## Tags vs. Groups

| Feature | Tags | Groups |
|---|---|---|
| **Purpose** | Categorize tasks for batch selection | Organize tasks with aggregate scoring |
| **Metrics** | No aggregate metrics | Can define `aggregate_metric_list` |
| **Display** | Tasks listed individually | Group appears as a row with subtasks underneath |
| **Nesting** | Flat (tags can't contain tags) | Hierarchical (groups can contain groups) |
| **Config** | Set in task YAML: `tag: [reasoning]` | Separate group YAML or inline |

**Use tags** when you just want to run a set of tasks together. **Use groups** when you need aggregate scores or hierarchical reporting.

## Basic group config

Create a YAML file with `group` and `task` keys:

```yaml
# lm_eval/tasks/nli/_nli.yaml
group: nli_tasks
task:
  - cb
  - anli_r1
  - rte
metadata:
  version: 1.0
```

This creates a group named `nli_tasks` containing three tasks. Running `lm-eval run --tasks nli_tasks` evaluates all three and shows them under a group header in the results table.

## Aggregate metrics

To report a single score for the group across all subtasks, add `aggregate_metric_list`:

```yaml
group: nli_tasks
task:
  - cb
  - anli_r1
  - rte
aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true   # micro-average (default)
metadata:
  version: 1.0
```

### `aggregate_metric_list` fields

| Field | Type | Description |
|---|---|---|
| `metric` | str | Name of the metric to aggregate (all subtasks must report this metric) |
| `aggregation` | str | Aggregation function (currently only `mean` is supported) |
| `weight_by_size` | bool | `true` for micro-average (weight by number of docs per subtask), `false` for macro-average (equal weight per subtask). Default: `true` |
| `filter_list` | str or list | Which filter pipeline(s) to match when aggregating (e.g., `"strict-match"`). Default: `"none"` |

!!! tip
    **Micro vs. macro averaging**: MMLU uses micro-averaging — if one subject has 200 questions and another has 50, the larger subject contributes more to the final score. This is equivalent to concatenating all subtasks into one dataset. Set `weight_by_size: false` for macro-averaging (equal weight per subtask).

## Overriding subtask config

Apply config overrides to subtasks within a group:

```yaml
group: my_benchmark
task:
  - task: mmlu
    num_fewshot: 5          # Override few-shot count for all MMLU subtasks
  - task: hellaswag
    num_fewshot: 0
```

When the subtask is itself a group (like `mmlu`), the override propagates to all its children.

## Inline subtask definitions

Define new subtasks directly inside the group config:

```yaml
group: nli_and_mmlu
task:
  - group: nli_tasks
    task:
      - cb
      - anli_r1
      - rte
    aggregate_metric_list:
      - metric: acc
        aggregation: mean
        higher_is_better: true

  - task: mmlu
    num_fewshot: 2
```

This creates a nested structure: `nli_and_mmlu` contains the inline `nli_tasks` group and the existing `mmlu` group.

## Python class-based subtasks

For tasks implemented as Python classes, use `!function`:

```yaml
group: scrolls
task:
  - task: scrolls_qasper
    class: !function task.Qasper
  - task: scrolls_quality
    class: !function task.QuALITY
  - task: scrolls_narrativeqa
    class: !function task.NarrativeQA
```

## The `::` path syntax

Navigate nested groups using `::` on the CLI or in Python:

```bash
# Run only the anatomy subtask from MMLU
lm-eval run --tasks mmlu::mmlu_anatomy --model hf --model_args pretrained=gpt2

# Navigate deeper nesting
lm-eval run --tasks my_benchmark::nli_tasks::cb --model hf --model_args pretrained=gpt2
```

In Python:

```python
from lm_eval.tasks import TaskManager

tm = TaskManager()
loaded = tm.load(["mmlu::mmlu_anatomy"])
```

## Formats in groups

Apply [prompt formats](prompt_formats.md) to tasks within groups using the `@` suffix:

```yaml
group: my_benchmark
task:
  - task: subtask_a@mcqa
    dataset_path: ...
    doc_to_text: question
    doc_to_target: answer
    doc_to_choice: choices

  - task: subtask_b
    formats: generate
    dataset_path: ...
    doc_to_text: question
    doc_to_target: answer
    doc_to_choice: choices
```

## Display names

Use `group_alias` and `task_alias` to provide cleaner display names:

```yaml
group: mmlu
group_alias: "MMLU"
task:
  - task: mmlu_abstract_algebra
    task_alias: "Abstract Algebra"
  - task: mmlu_anatomy
    task_alias: "Anatomy"
```

This keeps the registered names unique while showing cleaner names in the results table.

## GroupConfig reference

| Field | Type | Description |
|---|---|---|
| `group` | str | Group name (used on CLI to invoke the group) |
| `group_alias` | str | Display name for the results table |
| `task` | list | List of task names, group names, or inline task/group configs |
| `aggregate_metric_list` | list | Metrics to aggregate across subtasks (see fields above) |
| `metadata` | dict | Arbitrary metadata. Use `version` for versioning, `num_fewshot` to override the displayed n-shot value |
