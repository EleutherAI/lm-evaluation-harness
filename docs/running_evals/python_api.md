# Python API

This guide covers programmatic usage of the evaluation harness in Python scripts and applications.

## Overview

The library provides three main ways to run evaluations programmatically:

| Function | Use Case |
|----------|----------|
| `simple_evaluate()` | Most common — accepts model name strings or LM objects |
| `EvaluatorConfig` | Config-based — load settings from YAML or dataclass |
| `evaluate()` | Low-level — full control over task dictionaries |

---

## Using `simple_evaluate()`

The `simple_evaluate()` function is the recommended entry point for most use cases.

### Basic Usage

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2,dtype=float32",
    tasks=["hellaswag", "arc_easy"],
    num_fewshot=5,
    batch_size=8,
    device="cuda:0",
)
```

### With a Pre-initialized Model

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

# Initialize model separately
lm = HFLM(pretrained="gpt2", batch_size=16)

results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["hellaswag"],
    num_fewshot=0,
)
```

### With External Tasks

```python
import lm_eval
from lm_eval.tasks import TaskManager

# Include custom task definitions
task_manager = TaskManager(include_path="/path/to/custom/tasks")

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["my_custom_task"],
    task_manager=task_manager,
)
```

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str or LM | Model name (e.g., "hf", "vllm") or LM instance |
| `model_args` | str or dict | Model constructor arguments |
| `tasks` | list[str] | Task names to evaluate |
| `num_fewshot` | int | Number of few-shot examples |
| `batch_size` | int or str | Batch size or "auto" |
| `device` | str | Device (cuda, cpu, mps) |
| `limit` | int or float | Limit examples per task |
| `repeats` | int | Number of repeated runs per sample (for self-consistency) |
| `log_samples` | bool | Save model inputs/outputs (default: `True`) |
| `task_manager` | TaskManager | For external tasks |
| `gen_kwargs` | dict | Generation arguments |
| `apply_chat_template` | bool or str | Use chat template |
| `system_instruction` | str | System prompt |
| `fewshot_as_multiturn` | bool | Multi-turn few-shot |

### Return Value

`simple_evaluate()` returns a dictionary with:

```python
{
    "results": {
        "task_name": {
            "metric_name,filter_name": value,
            "metric_name,filter_name_stderr": stderr_value,
        }
    },
    "configs": {...},      # Task configurations
    "versions": {...},     # Task versions
    "n-shot": {...},       # Few-shot counts
    "higher_is_better": {...},
    "n-samples": {...},
    "samples": {...},      # If log_samples=True
}
```

---

## Using `TaskManager.load()`

The `TaskManager.load()` method is the modern way to build task dictionaries. It supports name strings, config dicts, runtime overrides, and the `@format` / `::` path syntax.

### Basic Usage

```python
from lm_eval.tasks import TaskManager

tm = TaskManager()
loaded = tm.load(["hellaswag", "arc_easy"])

loaded["tasks"]      # {"hellaswag": Task, "arc_easy": Task}
loaded["groups"]     # {} (no groups requested)
loaded["group_map"]  # {}
```

### Loading Groups

When you request a group, `load()` expands it into its leaf tasks and tracks the group structure:

```python
loaded = tm.load(["mmlu"])

loaded["tasks"]      # {"mmlu_abstract_algebra": Task, "mmlu_anatomy": Task, ...}
loaded["groups"]     # {"mmlu": Group}
loaded["group_map"]  # {"mmlu": ["mmlu_abstract_algebra", "mmlu_anatomy", ...]}
```

### Runtime Overrides

Override task config fields at load time without modifying YAML files:

```python
loaded = tm.load(
    ["mmlu", "arc_easy"],
    overrides={
        "arc_easy": {"num_fewshot": 5},
        "mmlu": {"num_fewshot": 3},
    },
)
```

Overrides apply to the named task or group. For groups, overrides propagate to all child tasks at evaluation time. Note that `group_map` itself only records each group's direct children — recursive propagation is handled by the evaluator.

### Inline Config Dicts

You can mix name strings with full config dicts:

```python
loaded = tm.load([
    "hellaswag",                    # by name
    {                               # inline config
        "task": "my_task",
        "dataset_path": "my_org/my_data",
        "test_split": "test",
        "doc_to_text": "question",
        "doc_to_target": "answer",
    },
])
```

### Return Type: `TaskDict`

`TaskManager.load()` returns a `TaskDict` TypedDict:

| Key | Type | Description |
|---|---|---|
| `tasks` | `dict[str, Task]` | Flat mapping of every leaf task name to its Task object |
| `groups` | `dict[str, Group]` | Flat mapping of every group name to its Group object |
| `group_map` | `dict[str, list[str]]` | Each group's direct children (not recursive) |

---

## Using `EvaluatorConfig`

The `EvaluatorConfig` class provides a structured way to manage evaluation settings.

### From YAML File

```python
from lm_eval.config.evaluate_config import EvaluatorConfig
import lm_eval

# Load configuration from YAML
config = EvaluatorConfig.from_config("eval_config.yaml")

# Process tasks
task_manager = config.process_tasks()

# Run evaluation
results = lm_eval.simple_evaluate(
    model=config.model,
    model_args=config.model_args,
    tasks=config.tasks,
    num_fewshot=config.num_fewshot,
    batch_size=config.batch_size,
    device=config.device,
    task_manager=task_manager,
    log_samples=config.log_samples,
    gen_kwargs=config.gen_kwargs,
    apply_chat_template=config.apply_chat_template,
    system_instruction=config.system_instruction,
)
```

### Direct Instantiation

```python
from lm_eval.config.evaluate_config import EvaluatorConfig

config = EvaluatorConfig(
    model="hf",
    model_args={"pretrained": "gpt2", "dtype": "float32"},
    tasks=["hellaswag", "arc_easy"],
    num_fewshot=5,
    batch_size=8,
    device="cuda:0",
    output_path="./results/",
    log_samples=True,
)

task_manager = config.process_tasks()
```

See the [Configuration Guide](config_files.md#config-schema) for all available fields.

---

## Using `evaluate()`

The `evaluate()` function provides lower-level control, accepting pre-built task dictionaries.

```python
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM

# Initialize model
lm = HFLM(pretrained="gpt2", batch_size=16)

# Build task dictionary using TaskManager.load()
tm = TaskManager()
loaded = tm.load(["hellaswag", "my_custom_task"])

# Run evaluation
results = lm_eval.evaluate(
    lm=lm,
    task_dict=loaded,
    num_fewshot=5,
    limit=100,
)
```

---

## Custom Models

To evaluate a custom model, create a subclass of `lm_eval.api.model.LM`:

```python
from lm_eval.api.model import LM

class MyCustomLM(LM):
    def __init__(self, model, batch_size=1):
        super().__init__()
        self.model = model
        self._batch_size = batch_size

    def loglikelihood(self, requests):
        # Return list of (logprob, is_greedy) tuples
        ...

    def generate_until(self, requests):
        # Return list of generated strings
        ...

    def loglikelihood_rolling(self, requests):
        # Return list of (logprob, is_greedy) tuples
        ...

    @property
    def batch_size(self):
        return self._batch_size
```

Then use it with `simple_evaluate()`:

```python
my_model = load_my_model()
lm = MyCustomLM(model=my_model, batch_size=16)

results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["hellaswag"],
)
```

For detailed guidance on implementing custom models, see the [Custom Model Backend](../extending/custom_model.md) guide.

---

## Logging

Configure logging for debugging:

```python
from lm_eval.utils import setup_logging

# Set log level
setup_logging("DEBUG")  # DEBUG, INFO, WARNING, ERROR

# Or use environment variable
import os
os.environ["LMEVAL_LOG_LEVEL"] = "DEBUG"
```

---

## Examples

### Batch Evaluation of Multiple Models

```python
import lm_eval

models = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
]

all_results = {}
for model_name in models:
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=["hellaswag"],
        batch_size="auto",
    )
    all_results[model_name] = results["results"]
```

### Save and Load Results

```python
import json
import lm_eval
from lm_eval.utils import handle_non_serializable

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["hellaswag"],
)

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, default=handle_non_serializable, indent=2)
```
