# Prompt Formats Guide

Formats let you control **how prompts are assembled** for evaluation tasks — instruction, question layout, choice labeling, answer solicitation — without writing Jinja templates by hand.

## Quick Start

### 1. Use a built-in format in your task YAML

```yaml
task: my_mcqa_task
dataset_path: my_org/my_dataset
test_split: test
doc_to_text: question
doc_to_target: answer
doc_to_choice: choices
formats: mcqa          # ← just add this line
```

The `formats` field tells lm-eval to apply the `mcqa` format, which auto-generates `doc_to_text`, `doc_to_target`, `doc_to_choice`, `output_type`, delimiters, and scoring — all from your three `doc_to_*` field mappings.

### 2. Or apply a format at runtime with `@`

No YAML changes needed — append `@format_name` to the task on the CLI:

```bash
lm_eval --tasks my_task@mcqa --model_args ...
lm_eval --tasks my_task@generate --model_args ...
lm_eval --tasks my_task@cloze --model_args ...
```

This is the fastest way to try different prompt styles on the same underlying dataset.

---

## Built-in Formats

| Name           | `output_type`     | Best for                                            | Choice labels   |
|----------------|-------------------|-----------------------------------------------------|-----------------|
| **`mcqa`**     | `multiple_choice` | Standard A/B/C/D benchmarks (ARC, MMLU, HellaSwag)  | `A. B. C. D.`   |
| **`cloze`**    | `multiple_choice` | Cloze-style / unlabeled loglikelihood scoring       | None (raw text) |
| **`generate`** | `generate_until`  | Open-ended generation with letter-answer extraction | `A. B. C. D.`   |
| **`cot`**      | `generate_until`  | Chain-of-thought reasoning, free-form answer        | None            |

### What each format produces

**`mcqa`** — classic multiple-choice:

```
Question: What is the capital of France?
A. Berlin
B. Madrid
C. Paris
D. London
Answer: C
```

**`cloze`** — loglikelihood over raw choice text (no labels):

```
Question: What is the capital of France?
Answer: Paris
```

**`generate`** — generation with structured answer extraction:

```
Given the following question and 4 candidate answers (A, B, C and D), choose the best answer.
Question: What is the capital of France?
A. Berlin
B. Madrid
C. Paris
D. London
Your response should end with "The best answer is [answer_letter]" where the [answer_letter] is one of A, B, C or D.
The best answer is C
```

**`cot`** — chain-of-thought generation:

```
Given the following problem, reason step by step to find the final answer.
Problem: What is the capital of France?
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.
```

---

## How It Works

A format **consumes** your `doc_to_text`, `doc_to_target`, and `doc_to_choice` field mappings and **produces** Jinja templates plus config overrides (`output_type`, `target_delimiter`, `fewshot_delimiter`, `scorer`, etc.) that are applied to the task config automatically.

```
Your YAML fields:                  Format generates:
─────────────────                  ─────────────────
doc_to_text: question       →     doc_to_text:   (full Jinja prompt template)
doc_to_target: answer       →     doc_to_target: (Jinja target template)
doc_to_choice: choices      →     doc_to_choice: (Jinja choice template)
formats: mcqa               →     output_type, target_delimiter, scorer, ...
```

Your `doc_to_*` fields are **consumed as inputs** — they tell the format which dataset columns to reference. The format then overwrites them with fully-rendered Jinja templates.

---

## Customizing a Format

### Override specific fields inline

Pass a dict with `type` plus any fields you want to override:

```yaml
formats:
  type: mcqa
  instruction: "Choose the correct answer for this science question.\n"
  question_prefix: "Q: "
  answer_prompt: "The answer is:"
```

### All configurable fields

| Field                | Description                                      | Default (mcqa) |
|----------------------|--------------------------------------------------|----------------|
| `instruction`        | Text prepended to every prompt                   | `null`         |
| `question_prefix`    | Label before the question                        | `"Question: "` |
| `choice_labels`      | `"letters"`, `"numbers"`, custom list, or `null` | `"letters"`    |
| `choice_delimiter`   | Separator between choices                        | `"\n"`         |
| `section_separator`  | Separator between prompt sections                | `"\n"`         |
| `answer_instruction` | Optional CoT instruction before answer prompt    | `null`         |
| `answer_prompt`      | Text soliciting the answer                       | `"Answer:"`    |
| `gen_prefix`         | Constrained-decoding prefix (generation only)    | `null`         |
| `target_delimiter`   | Separator between prompt and target in few-shot  | `" "`          |
| `fewshot_delimiter`  | Separator between few-shot examples              | `"\n\n"`       |
| `scorer`             | Scoring method name or config                    | `null`         |

### Example: numbered choices with custom instruction

```yaml
formats:
  type: mcqa
  instruction: "Select the correct option.\n\n"
  choice_labels: numbers      # 1. 2. 3. 4. instead of A. B. C. D.
  answer_prompt: "Option:"
```

### Example: custom choice labels

```yaml
formats:
  type: mcqa
  choice_labels: ["I", "II", "III", "IV"]
```

---

## Multi-Format Tasks

Override multiple formats in one YAML, then select at runtime with `@`:

```yaml
task: my_task
dataset_path: my_org/my_dataset
test_split: test
doc_to_text: question
doc_to_target: answer
doc_to_choice: choices
formats:
  mcqa:
    instruction: "Pick the right answer."
  generate:
    instruction: "Generate the answer."
```

Then run either variant:

```bash
lm_eval --tasks my_task@mcqa ...
lm_eval --tasks my_task@generate ...
```

When no `@suffix` is given, the **first key** is used as the default (here, `mcqa`).

---

## Using `@` With Any Task
# TODO: should probably not allow
The `@` suffix works even on tasks that **don't** declare a `formats:` field — the suffix is resolved against the global format registry:

```bash
# These work on any task that has doc_to_choice set:
lm_eval --tasks arc_easy@mcqa ...
lm_eval --tasks arc_easy@cloze ...
lm_eval --tasks arc_easy@generate ...
```

The task just needs `doc_to_text`, `doc_to_target`, and (for choice-based formats) `doc_to_choice` to be set in its YAML.

---

## Formats in Groups

Formats work in group configs too:

```yaml
group: my_benchmark
task:
  - task: subtask_a@mcqa
    dataset_path: ...
    doc_to_text: question
    doc_to_target: answer
    doc_to_choice: choices

  - task: subtask_b
    dataset_path: ...
    doc_to_text: question
    doc_to_target: answer
    doc_to_choice: choices
    formats: generate
```

---

## Jinja Variables in Format Fields

When `choice_labels` and `doc_to_choice` are both set, formats inject computed Jinja variables you can reference in `instruction` and `answer_prompt`:

| Variable                 | Example value          | Description              |
|--------------------------|------------------------|--------------------------|
| `{{ _num_choices }}`     | `4`                    | Number of choices        |
| `{{ _choice_labels }}`   | `['A', 'B', 'C', 'D']` | List of label strings    |
| `{{ _choice_list_and }}` | `A, B, C and D`        | Labels joined with "and" |
| `{{ _choice_list_or }}`  | `A, B, C or D`         | Labels joined with "or"  |

These are how the built-in `generate` format produces dynamic instructions like:

```yaml
instruction: "Given the following question and {{ _num_choices }} candidate answers ({{ _choice_list_and }}), choose the best answer.\n"
answer_prompt: 'Your response should end with "The best answer is [answer_letter]" where the [answer_letter] is one of {{ _choice_list_or }}.'
```

---

## Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IN YAML:                                                   │
│    formats: mcqa              # simple                      │
│    formats:                   # with overrides              │
│      type: mcqa                                             │
│      instruction: "..."                                     │
│    formats:                   # multi-format                │
│      mcqa: null                                             │
│      generate:                                              │
│        instruction: "..."                                   │
│                                                             │
│  ON CLI:                                                    │
│    --tasks my_task@mcqa       # runtime format selection    │
│    --tasks my_task@generate   # try a different format      │
│    --tasks my_task@cot        # chain-of-thought            │
│                                                             │
│  BUILT-IN FORMATS:                                          │
│    mcqa     → loglikelihood, A/B/C/D labels                 │
│    cloze    → loglikelihood, no labels                      │
│    generate → free generation, letter answer extraction     │
│    cot      → free generation, step-by-step reasoning       │
│                                                             │
│  YOUR TASK YAML NEEDS:                                      │
│    doc_to_text: <question field>                            │
│    doc_to_target: <answer field>                            │
│    doc_to_choice: <choices field>   (for mcqa/generate)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
