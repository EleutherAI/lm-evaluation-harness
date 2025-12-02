# Task Configuration Improvements for lm-evaluation-harness
---

## Overview

We've added two (maybe three) features that dramatically simplify task configuration:

1. **Task Templates** – Predefined prompt formats (MMLU, Cloze, Generate) that handle common evaluation patterns
   - test with `lm-eval run --model dummy --tasks arc_template --write_out --template mmlu` (or `--template cloze`)
     (defined [here](../lm_eval/tasks/template_examples/arc_easy.yaml))
2. **Task List** – Define multiple task variants in a single YAML file with a shared configuration
3. Hierarchical Groups. (Somewhat undercooked, currently)

---

## 1. Task Templates

### The Problem

Writing a new multiple-choice benchmark task currently requires specifying the full prompt format manually:

```yaml
task: my_benchmark
dataset_path: my_org/dataset
output_type: multiple_choice
doc_to_text: |
  Question: {{question}}
  A. {{choices[0]}}
  B. {{choices[1]}}
  C. {{choices[2]}}
  D. {{choices[3]}}
  Answer:
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{[A, B, C, D]}}"
```

To create a variant, have to create a new YAML file:

```yaml
task: my_benchmark_cloze
include: my_benchmark.yaml # or paste all the other common fields
doc_to_text: |
  Question: {{question}}
  Answer:
doc_to_target: "{{choices.label.index(answerKey)}}" # have to look up columns on HF manually
doc_to_choice: "{{choices.text}}"
```

Most benchmarks tend to rewrite this formatting logic.

### The Solution: `template:` Field

```yaml
# NEW: Clean, consistent, standard
task: my_benchmark
dataset_path: my_org/dataset
output_type: multiple_choice
template: mmlu                     # defined and hardcoded in backend
# enforced types for injective fields!!
doc_to_text: "{{question}}"        # Raw question only # [str]
doc_to_target: "{{answer}}"        # Index only # [int]
doc_to_choice: "{{choices}}"       # List only # [list[str]]
```

The template handles all the formatting:
- Adding "Question: " prefix
- Formatting choices as "A. choice1\nB. choice2..."
- Adding "\nAnswer:" suffix
- Proper delimiters between fewshot examples

### Available Templates

| Template | Format | Use Case |
|----------|--------|----------|
| `mmlu` | `Question: {q}\nA. {c0}\nB. {c1}...\nAnswer:` | Standard multiple choice |
| `cloze` | `Question: {q}\nAnswer:` | Fill-in-the-blank (choices scored by loglikelihood) |
| `generate_until` | `Question: {q}\nA. {c0}...\nAnswer:` | Generative MC (model outputs letter) |

### User API

#### Custom Template Parameters

```yaml
task: my_custom_task
template: #(templates?) maybe TODO
  mmlu:
      prefix: "Q: "                 # Override "Question: " prefix
      suffix: "\nA:"                # Override "\nAnswer:" suffix
      choice_delimiter: " | "       # Override "\n" between choices
      choice_format: numbers        # Use "1. 2. 3." instead of "A. B. C."
# modify more than one
  cloze:
      choice_format: numbers        # Still append 1. to choices in cloze format
```

#### Template Protocol (for custom templates)

```python
class Template(Protocol):
    format: str
    prefix: str                       # Before question
    suffix: str                       # After choices
    question_choice_delimiter: str    # Between question and choices
    choice_delimiter: str             # Between choices
    target_delimiter: str             # After answer in fewshots

    def format_prompt(self, q, c, a) -> str: ...
    def format_choices(self, q, c, a) -> list[str]: ...
    def format_target(self, q, c, a) -> str | int: ...
```

### CLI Experience
- either `--template cloze` or `task@cloze` (one! of either)

### Questions

- What fields should be named, and what should they be? Should we allow doc_to_text, etc within the field? example:

```yaml
template: #(templates?) maybe TODO
  mmlu:
    question: {question} # will map to doc_to_text
    answer: {answer}     # will map to doc_to_target
    choices: {choices}   # will map to doc_to_choice
```

Should we use the doc_to_* nomenclature here as well, or is this more clear

- Currently the templates only work one way (MC -> generative), as generation tasks tend not to have choices and increasingly also have their own process results.
How to extend this to purely generative tasks? For example minerva_math, minerva_math_cot, minerva_math_cot_thinking.

I have been meeaning to add a `chat_override` field as using the same for both base and chat is a bit buggy. But this would only overloads the fields, and would need to be called explicitly (`--chat_override`), or alternatively when `apply_chat_template`

### Implementation Location

- `lm_eval/config/template.py` – Template classes (`MMLUTemplate`, `ClozeTemplate`, `GenerateTemplate`)
- `lm_eval/config/task.py:188` – `template` field in TaskConfig
- `lm_eval/api/task.py:355` – `apply_template_format()` method

---

## 2. Task List

### The Problem

Creating multilingual or other similar benchmarks few-shot variants requires duplicating configuration across many files:

The usual pattern is to create a common config:

```yaml
tag:
  - hellaswag_multilingual
dataset_path: null
dataset_name: null
output_type: multiple_choice
training_split: null
validation_split: validation
test_split: null
process_docs: !function utils.process_docs
doc_to_text: "query"
doc_to_target: "{{label.lstrip()}}"
doc_to_choice: "choices"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

and then:

```
hellaswag/
├── hellaswag_en.yaml  
├── hellaswag_es.yaml  
├── hellaswag_fr.yaml  
├── hellaswag_de.yaml  
└── ... (20 more files)
```

of the form:

```yaml
include: _hellaswag_yaml
task: hellaswag_ar
dataset_path: alexandrainst/m_hellaswag
dataset_name: ar
training_split: null
validation_split: val

```

### The Solution: `task_list:` Field

```yaml
task_list:
  - task: variant_name_1
    override_field: value
  - task: variant_name_2
    override_field: different_value

# Common fields below
shared_field: value
```

```yaml
# hellaswag_multilingual.yaml - ONE file for all variants
task_list:
  - task: hellaswag_en
    dataset_name: en
  - task: hellaswag_es
    dataset_name: es
    num_fewshot: 5  
  - task: hellaswag_fr
    dataset_name: fr
  - task: hellaswag_de
    dataset_name: de

# Shared configuration (applies to all tasks above)
dataset_path: hellaswag_multilingual
output_type: multiple_choice
template: mmlu
doc_to_text: "{{question}}"
doc_to_target: "{{answer}}"
metric_list:
  - metric: acc
```

#### With Include

```yaml
include: base_task.yaml      # Import base configuration

task_list:
  - task: arc_0shot
    num_fewshot: 0
  - task: arc_5shot
    num_fewshot: 5
  - task: arc_25shot
    num_fewshot: 25
```

### CLI Experience

Tasks from `task_list` appear and will be registered as as **regular standalone tasks**.

### How Config Merging Works

When you request `hellaswag_es`:

1. System finds `hellaswag_es` in the task index
2. Loads the YAML containing `task_list`
3. Finds the entry `{task: hellaswag_es, dataset_name: es, num_fewshot: 5}`
4. Merges: `final_config = {**common_fields, **task_specific_overrides}`
5. Result: Full config with `dataset_name: es` and `num_fewshot: 5`

### Key Differences from Groups

| Aspect               | Task List             | Groups                        |
|----------------------|-----------------------|-------------------------------|
| Creates group object | No                    | Yes                           |
| Tasks are standalone | Yes                   | No (nested)                   |
| Direct task access   | `hellaswag_en`        | No                            |
| Use case             | Variants of same task | Organizing related benchmarks |

---

## 3. Hierarchical Tasks (`children:` syntax)

- Currently, we allow some overloading when defining groups, but it's an either or in the sense we cannot call a subgroup or task solely defined in a group config. The way around is to create individual configs for each subgroup/task, for example for mmlu, we have `_mmlu_stem`, `mmlu_other`, defined in _different_ configs and then we import them into the main `mmlu` group. This is repeated for each benchmark variant, mmlu_continuation, mmlu_flan_zero_shot etc. One solution could be in allowing hierarchical groups:

### User API

```yaml
group: my_benchmark
children:
  easy:
    dataset_path: my_org/benchmark
    dataset_name: easy
    template: mmlu
    doc_to_text: "{{question}}"

  hard:
    dataset_path: my_org/benchmark
    dataset_name: hard
    template: mmlu
    doc_to_text: "{{question}}"

aggregate_metric_list:
  - metric: acc
    weight_by_size: true
```

#### CLI with `::` path addressing

```bash
lm_eval --tasks my_benchmark           # Run all
lm_eval --tasks my_benchmark::easy     # Run only easy
lm_eval --tasks my_benchmark::hard     # Run only hard
```

#### Nested groups

```bash
lm_eval --tasks mmlu::stem::math       # Specific task
lm_eval --tasks mmlu::stem             # All STEM tasks
```

#### Nested Group Example

```yaml
group: nested_group
children:
  subgroup_a:
    children:
      task_1:
        dataset_path: json
        dataset_kwargs:
          data_files:
            test: tests/test_configs/test_data.json
        output_type: multiple_choice
        doc_to_text: "{{question}}"
        doc_to_target: "{{choices[answer]}}"
        test_split: test
        metric_list:
          - metric: acc

      task_2:
        dataset_path: json
        # ... same structure

  subgroup_b:
    children:
      task_3:
        dataset_path: json
        # ... same structure

aggregate_metric_list:
  - metric: acc
    weight_by_size: true
```

#### Referencing Existing Tasks

```yaml
group: my_suite
children:
  # Inline task definition
  custom_task:
    dataset_path: ...

  # Reference existing task
  arc:
    ref: arc_easy

  # Reference existing group
  hellaswag:
    ref: hellaswag

  # Expand all tasks with a tag
  arc_tasks:
    tag: ai2_arc
```

## Question:

Is this a good pattern? How should we handle groups vs. templates? I was thinking `@` for templates, and then you could do:

- arc_easy@mmlu
- mmlu::stem@cloze

Alternatively, `::` would work with tasks (as they are leaf nodes), but will become a mess parsing when it comes to groups, or tags.

Should `task_list` or `tags` also be namedscaped? Consider: `mmlu_flan_cot_fewshot_astronomy`.

## Migration Guide

### Before (verbose)

```yaml
# arc_easy_0shot.yaml
task: arc_easy_0shot
include: _arc_base.yaml
num_fewshot: 0
doc_to_text: |
  Question: {{question}}
  A. {{choices.text[0]}}
  B. {{choices.text[1]}}
  ...
```

### After (clean)

```yaml
# arc_variants.yaml
task_list:
  - task: arc_easy_0shot
    num_fewshot: 0
  - task: arc_easy_5shot
    num_fewshot: 5

include: _arc_base.yaml
template: mmlu
doc_to_text: "{{question}}"
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{choices.text}}"
```

---
