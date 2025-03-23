# LM Evaluation Harness

A framework for evaluating language models developed by EleutherAI.
    
## Overview

The LM Evaluation Harness is designed to facilitate the integration of various API-based language models into a standardized evaluation framework. This tool allows researchers and developers to:

- Evaluate model performance on a wide range of tasks
- Compare different models using consistent metrics
- Extend the framework with custom tasks and models

## Installation

```bash
# Basic installation
pip install lm-eval

# With additional dependencies
pip install "lm-eval[gptq,vllm]"

# For development
pip install -e ".[dev]"
```

## Quick Start

```python
# Basic usage example
import lm_eval

results = lm_eval.simple_evaluate(
    model="gpt2",
    tasks=["hellaswag", "mmlu"],
    num_fewshot=0
)
```

## Command Line Usage

```bash
lm-eval --model hf --model_args pretrained=gpt2 --tasks hellaswag --num_fewshot 0
```

## Features

- Support for evaluating text-only and multimodal models
- Flexible API for integrating custom models and tasks
- Comprehensive benchmarking capabilities
- Caching mechanisms for faster evaluation
- Extensible framework for adding new tasks and evaluation metrics

## Documentation Guide

Welcome to the docs for the LM Evaluation Harness! Here's what you'll find in our documentation:

* **[Interface](./interface.md)** - Learn about the public interface of the library, including how to evaluate via the command line or as integrated into an external library.
* **[Model Guide](./model_guide.md)** - Learn how to add a new library, API, or model type to the framework, with explanations of different evaluation approaches.
  * **[API Guide](./API_guide.md)** - Extended guide on how to extend the library to new model classes served over an API.
* **[New Task Guide](./new_task_guide.md)** - A crash course on adding new tasks to the library.
* **[Task Configuration Guide](./task_guide.md)** - Advanced documentation on pushing the limits of task configuration that the Eval Harness supports.
