# LM Evaluation Harness

A unified framework for evaluating generative language models on a large number of different evaluation tasks.

## Features

- **60+ benchmarks** with hundreds of subtasks, including MMLU, HellaSwag, GSM8K, ARC, and more
- **Multiple backends** — HuggingFace Transformers, vLLM, OpenAI-compatible APIs, and custom models
- **YAML-based configuration** with Jinja2 templating and [declarative prompt formats](writing_tasks/prompt_formats.md) — use `formats: mcqa` to auto-generate A/B/C/D prompts from simple field mappings, or try different formats at runtime with `--tasks my_task@generate`
- **Reproducible evaluations** with published prompts, versioning, and shareable configs
- **Extensible scoring** — pluggable scorers, metrics, and filter pipelines

## Quick start

```bash
pip install lm-eval[hf]
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag
```

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["hellaswag"],
)
```

See the [Quickstart guide](getting_started/quickstart.md) for a complete walkthrough.

## Documentation

| I want to... | Start here |
|---|---|
| Get up and running | [Quickstart](getting_started/quickstart.md) |
| Run evaluations from CLI or Python | [CLI Reference](running_evals/cli_reference.md) / [Python API](running_evals/python_api.md) |
| Create or customize evaluation tasks | [Your First Task](writing_tasks/your_first_task.md) |
| Use prompt formats to simplify task authoring | [Prompt Formats](writing_tasks/prompt_formats.md) |
| Add a model backend, scorer, or metric | [Custom Model](extending/custom_model.md) / [Custom Scorers](extending/custom_scorers.md) |
| Upgrade from v0.4 | [Migrating from v0.4](migration_v0_5.md) |
| Browse the API reference | [API Reference](api/index.md) |
| Contribute to the project | [Contributing](contributing.md) |
