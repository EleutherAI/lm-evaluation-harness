# LM Evaluation Harness

A unified framework for evaluating generative language models on a large number of different evaluation tasks.

## Features

- Over 60 standard academic benchmarks with hundreds of subtasks
- Support for HuggingFace Transformers, vLLM, SGLang, OpenAI APIs, and more
- YAML-based task configuration with Jinja2 templating
- Reproducible evaluations with published prompts

## Quick Start

### Installation

```bash
pip install lm-eval
pip install lm-eval[hf]   # For HuggingFace models
```

### Basic Evaluation

```bash
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag
```

### Python API

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["hellaswag"],
)
```

## Documentation

| Section | Description |
|---------|-------------|
| [User Guide](interface.md) | CLI reference and command-line usage |
| [Python API](python-api.md) | Programmatic usage in Python |
| [Configuration Files](config_files.md) | YAML configuration guide |
| [Model Guide](model_guide.md) | Adding new model backends |
| [New Task Guide](new_task_guide.md) | Creating evaluation tasks |
| [Task Configuration](task_guide.md) | Advanced YAML task config |
| [API Reference](api/index.md) | Auto-generated from docstrings |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |
