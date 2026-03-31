# Migration Guide: v0.4 → v0.5

This guide covers the key changes in v0.5 of the LM Evaluation Harness and how to update your code.

## Summary of changes

| Area | What changed |
|---|---|
| Task loading | `TaskManager.load()` replaces `get_task_dict()` / `load_task_or_group()` |
| Groups | `Group` dataclass replaces `ConfigurableGroup` |
| Scoring | New `Scorer` pipeline replaces direct metric processing |
| Metrics | `Metric[_T, _K]` generic dataclass, modular metric system |
| Formats | New declarative prompt format system (additive, not breaking) |
| Instances | Generic `Instance[InputT, OutputT]` with typed aliases |
| Config | `TaskConfig` is now a proper dataclass with TypedDict configs |

## Task loading

### Before (v0.4)

```python
from lm_eval.tasks import get_task_dict, TaskManager

task_manager = TaskManager()
task_dict = get_task_dict(["hellaswag", "arc_easy"], task_manager)
# task_dict is a nested dict with ConfigurableGroup keys
```

### After (v0.5)

```python
from lm_eval.tasks import TaskManager

tm = TaskManager()
loaded = tm.load(["hellaswag", "arc_easy"])

loaded["tasks"]      # {"hellaswag": Task, "arc_easy": Task}
loaded["groups"]     # {}
loaded["group_map"]  # {}
```

**Key differences:**

- `load()` returns a `TaskDict` TypedDict with flat `tasks`, `groups`, and `group_map` keys
- Supports runtime overrides: `tm.load(["arc_easy"], overrides={"arc_easy": {"num_fewshot": 5}})`
- Supports inline config dicts and the `@format` / `::` path syntax
- `get_task_dict()` and `load_task_or_group()` are deprecated but still work

## Groups

### Before (v0.4)

```python
from lm_eval.api.group import ConfigurableGroup

# ConfigurableGroup with custom __eq__/__hash__ based on group_name
# Used as dict keys in nested task dicts
```

### After (v0.5)

```python
from lm_eval.api.group import Group

# Group is a @dataclass — simpler, no custom __eq__/__hash__
# Returned in TaskDict["groups"] as a flat dict, not as dict keys
```

`ConfigurableGroup` is deprecated. `Group` is a straightforward dataclass.

## Scoring pipeline

### Before (v0.4)

Metrics were processed directly in the evaluator via `Task.process_results()` and `Task.aggregation()`. Filter pipelines were managed separately.

### After (v0.5)

The `Scorer` class encapsulates the full pipeline: **filter → score → reduce → aggregate**.

```python
from lm_eval.scorers import Scorer, GenScorer, LLScorer, build_scorer
```

Scorer hierarchy:

- `Scorer` — abstract base
- `GenScorer` — for `generate_until` tasks (with 3 extensibility tiers)
- `LLScorer` — for `loglikelihood` / `multiple_choice` tasks
- Built-in scorers: `ChoiceMatchScorer`, `FirstTokenScorer`, `RegexExtractionScorer`

**For YAML task authors**: No changes needed for simple tasks. The task YAML fields (`metric_list`, `filter_list`) still work and are automatically routed through the scorer system.

**For Python API users**: If you were calling `Task.process_results()` directly, use the scorer pipeline instead.

## Metrics

### Before (v0.4)

```python
# Monolithic lm_eval/api/metrics.py
from lm_eval.api.metrics import register_metric, register_aggregation
```

### After (v0.5)

```python
# Modular lm_eval/api/metrics/ package
from lm_eval.api.metrics import Metric, register_metric, register_aggregation
```

The `Metric` class is now a generic frozen dataclass:

```python
@dataclass(frozen=True)
class Metric(Generic[_T, _K]):
    name: str
    fn: MetricFn[_T]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    aggregation: AggregationFn[_K] | None = None
    higher_is_better: bool = True
    output_type: set[str] = field(default_factory=lambda: {"multiple_choice"})
    reduction: ReductionFn[_T, _K] | None = take_first
```

Type chain: `fn(...) → _T`, `reduction(...) → _K`, `aggregation(Sequence[_K]) → float`.

The metrics module is now split into sub-modules: `metric.py`, `aggregations.py`, `corpus.py`, `generation.py`, `ll.py`, `reduce.py`, `stderr.py`, `utils.py`.

**For most users**: `register_metric` and `register_aggregation` still work the same way.

## Formats (new feature)

The declarative prompt format system is entirely new — no migration needed, but it's one of the biggest quality-of-life improvements in v0.5. Formats let task authors skip Jinja entirely for common prompt patterns.

**Before (v0.4)** — manual Jinja for a standard MCQA task:

```yaml
output_type: multiple_choice
doc_to_text: "Question: {{question}}\n{% for letter, choice in zip(['A','B','C','D'], choices) %}{{letter}}. {{choice}}\n{% endfor %}Answer:"
doc_to_target: "{{['A','B','C','D'][answer]}}"
doc_to_choice: "{{choices}}"
```

**After (v0.5)** — same result with a format:

```yaml
doc_to_text: question
doc_to_target: answer
doc_to_choice: choices
formats: mcqa
```

The format auto-generates the Jinja templates, sets `output_type`, configures delimiters, and wires up scoring.

**Runtime format selection** — try different prompt styles without touching YAML:

```bash
lm-eval run --tasks my_task@mcqa --model hf --model_args pretrained=gpt2
lm-eval run --tasks my_task@generate --model hf --model_args pretrained=gpt2
lm-eval run --tasks my_task@cloze --model hf --model_args pretrained=gpt2
```

Built-in formats: `mcqa`, `cloze`, `generate`, `cot`. See [Prompt Formats](writing_tasks/prompt_formats.md) for the full guide.

## Instance types

### Before (v0.4)

```python
from lm_eval.api.instance import Instance

# Instance with metadata as a tuple: (task_name, doc_id, repeats)
inst = Instance(
    request_type="loglikelihood",
    doc=doc,
    arguments=("context", "continuation"),
    metadata=(task_name, doc_id, repeats),
)
```

### After (v0.5)

```python
from lm_eval.api.instance import Instance, LLInstance, GenInstance

# Instance is generic with explicit fields
inst = Instance(
    request_type="loglikelihood",
    doc=doc,
    arguments=("context", "continuation"),
    task_name=task_name,
    doc_id=doc_id,
    repeats=repeats,
    target=gold_answer,
)
```

**Key changes:**

- `Instance[InputT, OutputT]` is now generic
- `task_name`, `doc_id`, `repeats` are explicit fields (not packed in a metadata tuple)
- New `target` field for gold references
- New `additional_args` field for multimodal support
- `metadata` is now a `dict[str, Any]` (not a tuple)
- Type aliases: `LLInstance`, `GenInstance`
- Backward compatibility: if you pass a tuple as `metadata`, `__post_init__` unpacks it

## Type aliases

New type aliases in `lm_eval.api._types`:

| Type | Definition |
|---|---|
| `Doc` | `dict[str, Any]` |
| `DataSplit` | `datasets.Dataset \| Sequence[Doc]` |
| `Dataset` | `Mapping[str, DataSplit] \| datasets.DatasetDict` |
| `Context` | `str \| list[dict[str, str]]` |
| `LLArgs` | `tuple[str, str]` |
| `LLOutput` | `tuple[float, bool]` |
| `GenArgs` | `tuple[Context, GenKwargs]` |
| `Completion` | `str` |
| `Reference` | `str \| list[str] \| int \| list[int] \| None` |

## TaskConfig

`TaskConfig` is now a proper dataclass (in `lm_eval.config.task`) with typed config dictionaries:

- `MetricConfig` — TypedDict for metric entries in `metric_list`
- `FilterStep` — TypedDict for filter steps in `filter_list`
- `ScorerConfig` — TypedDict for scorer configuration
- `FilterPipeline` — A named pipeline with filters and optional per-pipeline metrics

## What still works from v0.4

- YAML task configs are fully backward compatible
- `register_metric`, `register_aggregation`, `register_filter`, `register_model` decorators work the same
- `simple_evaluate()` API is unchanged
- `--apply_chat_template`, `--fewshot_as_multiturn`, `--system_instruction` work the same
- All existing task YAMLs in `lm_eval/tasks/` continue to work without modification
