# Plugin Guide

`lm-eval` resolves components — model backends, filters, metrics, aggregations — by
name against internal registries. A plugin is a component that lives in **your own
package** and gets registered into those registries at runtime, so `lm-eval` treats
it exactly like a built-in without you forking or editing the `lm-eval` source tree.

## What can be contributed

| Component   | Entry-point group      | Decorator               | Resolved by          |
|-------------|------------------------|-------------------------|----------------------|
| Model       | `lm_eval.models`       | `@register_model`       | `--model <name>`     |
| Filter      | `lm_eval.filters`      | `@register_filter`      | `filter_list` in a task YAML |
| Metric      | `lm_eval.metrics`      | `@register_metric`      | `metric_list` in a task YAML |
| Aggregation | `lm_eval.aggregations` | `@register_aggregation` | `aggregation` in a task YAML |

Tasks are *not* plugins — point `--include_path` at a directory of task YAMLs
instead. See the [New Task Guide](./new_task_guide.md).

## 1. Entry points (zero-config, recommended for published packages)

Declare the entry point in your package's `pyproject.toml`, mapping the name
`lm-eval` should resolve to `module:object`:

```toml
[project.entry-points."lm_eval.models"]
my-backend = "your_pkg.models:MyBackendLM"

[project.entry-points."lm_eval.metrics"]
my-metric = "your_pkg.metrics:my_metric"
```

Once your package is `pip install`ed it works with no extra flags:

```bash
lm-eval run --model my-backend --tasks hellaswag
```

Discovery is **lazy**: the entry point is recorded as a placeholder, and your module
is only imported when the component is actually requested. A broken plugin therefore
only breaks runs that ask for it, and is reported with the offending name.

Your object can also carry its `@register_*` decorator — importing the module fires
the decorator, which upgrades the placeholder in place. The two are reconciled
automatically as long as the entry-point value and the decorated object agree.

### Metrics need their decorator

`higher_is_better` and the default aggregation are recorded as *side effects* of
`@register_metric(...)`, not as separate entry points. So a metric plugin's module
must run the decorator, and the entry point should name the decorated function:

```python
# my_pkg/metrics.py
from lm_eval.api.registry import register_aggregation, register_metric


@register_aggregation("my-agg")
def my_agg(items):
    return sum(items) / len(items)


@register_metric(metric="my-metric", higher_is_better=True, aggregation="my-agg")
def my_metric(items):
    return sum(items)
```

If a metric's aggregation is also a plugin, register it in the **same module** — the
aggregation must already be in the registry when `@register_metric` runs.

## 2. Explicit import (for local or unpublished modules)

Point `lm-eval` at one or more modules to import before evaluation, so their
`@register_*` decorators run:

```bash
lm-eval run --model my-backend --plugins my_pkg.models --tasks hellaswag
```

`--plugins` accepts a comma-separated or space-separated list, and is also settable
as `plugins:` in a `--config` YAML. A module that fails to import is logged and
skipped rather than aborting the run.

## Precedence

Built-in components always win. Discovery skips any alias that is already
registered, so a plugin declaring `acc` cannot shadow the core `acc` metric — the
plugin module is not even imported. Pick a distinctive name.

## Programmatic use

Entry-point plugins need no wiring at all — they are discovered on first name
resolution, including from `lm_eval.simple_evaluate(...)`.

For a module that is not an installed entry-point plugin, import it yourself before
evaluating.

!!! Warning
Registration mutates **process-global** registries and is not scoped to
a single evaluation:

```python
import lm_eval
from lm_eval.api.registry import import_plugins

import_plugins(["my_pkg.models"])   # or simply: import my_pkg.models

results = lm_eval.simple_evaluate(model="my-backend", tasks=["hellaswag"])
```
