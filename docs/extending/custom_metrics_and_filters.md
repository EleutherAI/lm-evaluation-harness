# Custom Metrics & Filters

This guide covers implementing custom metrics, aggregation functions, and filters for the evaluation harness.

## Custom metrics

### The `Metric[_T, _K]` generic

The `Metric` dataclass is generic over two type parameters:

- `_T` — Per-sample result type from the metric function
- `_K` — Reduced type after collapsing repeats

The type chain flows: `fn(...) → _T`, `reduction(...) → _K`, `aggregation(Sequence[_K]) → float`.

For example, an accuracy metric has `_T = float` (0.0 or 1.0 per sample) and `_K = float` (after reduction), and the aggregation computes the mean.

### Registering a metric

Use the `@register_metric` decorator:

```python
from lm_eval.api.registry import register_metric

@register_metric(
    metric="my_metric",
    higher_is_better=True,
    output_type="generate_until",  # or "multiple_choice", etc.
    aggregation="mean",
)
def my_metric_fn(references, predictions, **kwargs):
    """Score a single document.

    Args:
        references: Gold-standard reference(s)
        predictions: Model prediction(s)
        **kwargs: Extra arguments from MetricConfig

    Returns:
        float: The metric score for this document
    """
    return 1.0 if predictions == references else 0.0
```

The decorator registers the metric name, its default aggregation, and whether higher is better. These can be overridden in YAML:

```yaml
metric_list:
  - metric: my_metric
    aggregation: median       # Override the registered default
    higher_is_better: false   # Override
```

### Using `!function` for one-off metrics

For metrics that don't need to be globally registered:

```python
# utils.py (in task directory)
def my_custom_metric(references, predictions, **kwargs):
    return 1.0 if predictions.strip() == references.strip() else 0.0
```

```yaml
metric_list:
  - metric: !function utils.my_custom_metric
    aggregation: mean
    higher_is_better: true
```

### Multiple-choice metrics

Multiple-choice metrics typically need both the gold label and the predicted label. Register them with `output_type="multiple_choice"`:

```python
@register_metric(
    metric="mcc",
    higher_is_better=True,
    output_type="multiple_choice",
    aggregation="matthews_corrcoef",
)
def mcc_fn(items):
    return items  # Passthrough — aggregation handles the actual computation
```

The aggregation function receives all `(gold, predicted)` pairs:

```python
@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items):
    unzipped = list(zip(*items))
    golds = unzipped[0]
    preds = unzipped[1]
    return sklearn.metrics.matthews_corrcoef(golds, preds)
```

## Custom aggregation functions

Register aggregation functions with `@register_aggregation`:

```python
from lm_eval.api.registry import register_aggregation

@register_aggregation("my_aggregation")
def my_aggregation(items):
    """Aggregate per-document scores into a single value.

    Args:
        items: Sequence of per-document metric values

    Returns:
        float: The aggregated score
    """
    return sum(items) / len(items)  # Simple mean
```

Then reference it in YAML:

```yaml
metric_list:
  - metric: exact_match
    aggregation: my_aggregation
```

## Custom reduction functions

Reduction functions collapse repeated runs per document (when `repeats > 1`):

```python
def my_reduction(references, predictions):
    """Reduce multiple predictions per document to one.

    Args:
        references: Gold reference (same for all repeats)
        predictions: List of predictions (one per repeat)

    Returns:
        The reduced prediction
    """
    # Example: return the most common prediction
    from collections import Counter
    counts = Counter(predictions)
    return counts.most_common(1)[0][0]
```

Reference in YAML:

```yaml
metric_list:
  - metric: exact_match
    reduction: !function utils.my_reduction
```

Built-in reductions: `take_first` (default), `mean`, `pass_at_k`.

## Custom filters

### Implementing a filter

Subclass `Filter` and implement the `apply` method:

```python
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("normalize_answer")
class NormalizeAnswerFilter(Filter):
    """Normalize answers by stripping whitespace and lowercasing."""

    def __init__(self, strip_punct=False):
        self.strip_punct = strip_punct

    def apply(self, resps, docs):
        """Apply the filter to model responses.

        Args:
            resps: list[list[str]] — responses grouped by document.
                   Each inner list has one response per repeat.
            docs: list[dict] — corresponding documents

        Returns:
            list[list[str]] — filtered responses, same shape
        """
        filtered = []
        for doc_resps in resps:
            normed = []
            for r in doc_resps:
                r = r.strip().lower()
                if self.strip_punct:
                    r = r.rstrip(".,;:!?")
                normed.append(r)
            filtered.append(normed)
        return filtered
```

### Using a custom filter

Reference it by its registered name in YAML:

```yaml
filter_list:
  - name: "clean-match"
    filter:
      - function: "normalize_answer"
        kwargs:
          strip_punct: true
      - function: "take_first"
```

### Filter contract

- **Input**: `list[list[response]]` — one inner list per document, each containing responses across repeats
- **Output**: Same shape — the filter transforms responses but preserves the per-document × per-repeat structure
- The final filter in a pipeline should reduce to one response per document (typically `take_first`)

## Putting it all together

A complete example combining custom metric, aggregation, and filter:

```python
# my_task/utils.py
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter, register_metric, register_aggregation

@register_filter("extract_number")
class ExtractNumberFilter(Filter):
    def apply(self, resps, docs):
        import re
        return [
            [re.search(r"(\d+)", r).group(1) if re.search(r"(\d+)", r) else ""
             for r in doc_resps]
            for doc_resps in resps
        ]

@register_metric(
    metric="numeric_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean",
)
def numeric_match(references, predictions, tolerance=0):
    try:
        ref = float(references)
        pred = float(predictions)
        return 1.0 if abs(ref - pred) <= tolerance else 0.0
    except (ValueError, TypeError):
        return 0.0
```

```yaml
# my_task/my_task.yaml
task: my_math_task
dataset_path: my_org/math_problems
test_split: test
output_type: generate_until
doc_to_text: "Solve: {{problem}}\nAnswer:"
doc_to_target: "{{answer}}"
generation_kwargs:
  until: ["\n"]
filter_list:
  - name: "numeric"
    filter:
      - function: "extract_number"
      - function: "take_first"
    metric_list:
      - metric: numeric_match
        tolerance: 0.01
```
