# Quickstart

Get from zero to your first evaluation results in under 5 minutes.

## Install

```bash
pip install lm-eval[hf]
```

This installs the harness plus the HuggingFace Transformers backend. See [Installation](installation.md) for other backends and options.

## Run your first evaluation

Evaluate GPT-2 on HellaSwag using the CLI:

```bash
lm-eval run \
    --model hf \
    --model_args pretrained=gpt2 \
    --tasks hellaswag \
    --limit 100
```

!!! tip
    `--limit 100` evaluates on only 100 samples for a quick test. Remove it for a full evaluation run.

Or from Python:

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["hellaswag"],
    limit=100,
)

print(results["results"])
```

## Reading the output

After an evaluation completes, you'll see a results table like:

```
|  Tasks   |Version|Filter|n-shot| Metric |Value |   |Stderr|
|----------|------:|------|-----:|--------|-----:|---|-----:|
|hellaswag |      1|none  |     0|acc     |0.2891|±  |0.0045|
|          |       |none  |     0|acc_norm|0.3108|±  |0.0046|
```

Key columns:

- **Tasks**: The evaluation task name
- **Filter**: Which output filter pipeline produced this result (e.g., `none` means no post-processing)
- **n-shot**: Number of few-shot examples used
- **Metric**: The scoring metric — `acc` is raw accuracy, `acc_norm` is length-normalized accuracy
- **Value**: The score (0–1 scale)
- **Stderr**: Standard error of the mean, computed via bootstrap

## Try different prompt formats

One of the most powerful features in lm-eval is **prompt formats** — they let you change how prompts are assembled without editing YAML. Just append `@format_name` to any task:

```bash
# Standard A/B/C/D multiple-choice
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag@mcqa --limit 100

# Cloze-style (no choice labels)
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag@cloze --limit 100

# Free generation with answer extraction
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag@generate --limit 100
```

Built-in formats: `mcqa`, `cloze`, `generate`, `cot` (chain-of-thought). See the [Prompt Formats guide](../writing_tasks/prompt_formats.md) for details.

## Explore available tasks

List all built-in tasks:

```bash
lm-eval ls tasks
```

List tasks matching a pattern:

```bash
lm-eval ls tasks --pattern "mmlu*"
```

## Where next?

| I want to... | Go to |
|---|---|
| Run evaluations with different models, settings, and options | [Running Evaluations](../running_evals/cli_reference.md) |
| Create a new evaluation task or customize an existing one | [Writing Tasks](../writing_tasks/your_first_task.md) |
| Add a new model backend or extend the scoring pipeline | [Extending the Framework](../extending/custom_model.md) |
| Understand the evaluation pipeline architecture | [Concepts & Architecture](concepts.md) |
