# Filters

Filters are post-processing steps applied to raw model outputs before scoring. They let you extract answers, clean text, subset responses, or ensemble over multiple generations — all configurable in YAML.

## How filters work

After the model runs on each `Instance`, raw responses are stored in `Instance.resps`. Filters process these responses before they reach the scoring pipeline:

```
Raw model responses → Filter pipeline → Filtered responses → Scorer
```

Filters operate on a list of responses per document. A single filter step transforms this list (e.g., apply a regex to each response), and a **pipeline** chains multiple steps together.

## Basic filter configuration

Use `filter_list` in your task YAML to define one or more filter pipelines:

```yaml
filter_list:
  - name: "get-answer"
    filter:
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
```

Each pipeline has:

- **`name`** — identifier for this pipeline (appears in results table)
- **`filter`** — ordered list of filter steps to apply

Each filter step (`FilterStep`) has:

- **`function`** — name of a registered filter (e.g., `regex`, `take_first`, `majority_vote`)
- **`kwargs`** — optional keyword arguments passed to the filter (can also be specified as flat keys alongside `function` — both forms are accepted)

## Built-in filters

A full list is available in `lm_eval/filters/__init__.py`. Common ones include:

| Filter | Description | Key kwargs |
|---|---|---|
| `noop` | No-op passthrough (identity) | — |
| `take_first` | Select only the first response per document | — |
| `take_first_k` | Select the first `k` responses | `k` |
| `regex` | Apply regex extraction to each response | `regex_pattern`, `group_select` |
| `remove_whitespace` | Strip whitespace from each response | — |
| `lowercase` | Lowercase each response | — |
| `majority_vote` | Return the most common response | — |
| `map` | Apply a Python function to each response | `mapping_dict` |
| `custom` | Apply a custom function | `filter_fn` |

## Multiple filter pipelines

Tasks can define multiple filter pipelines that run on **the same model outputs**. Each pipeline produces its own set of filtered responses and metric scores.

This is powerful for comparing different answer-extraction strategies or for self-consistency evaluation.

### Example: GSM8K with self-consistency

This task generates 64 chain-of-thought outputs per problem, then scores them three different ways:

```yaml
repeats: 64
filter_list:
  - name: "score-first"
    filter:
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
      - function: "take_first"

  - name: "maj@64"
    filter:
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
      - function: "majority_vote"
      - function: "take_first"

  - name: "maj@8"
    filter:
      - function: "take_first_k"
        k: 8
      - function: "regex"
        regex_pattern: "The answer is (\\-?[0-9\\.\\,]*[0-9]+)"
      - function: "majority_vote"
      - function: "take_first"
```

**`score-first`**: Extract the answer from the first generation only.

**`maj@64`**: Extract answers from all 64 generations, majority vote across them.

**`maj@8`**: Subset to the first 8 generations, then extract and majority vote.

All three pipelines produce separate metric rows in the results table — from one set of model outputs.

### Per-pipeline metrics

Each pipeline can specify its own metrics:

```yaml
filter_list:
  - name: "strict"
    filter:
      - function: "regex"
        regex_pattern: "(\\d+)"
      - function: "take_first"
    metric_list:
      - metric: exact_match

  - name: "flexible"
    filter:
      - function: "take_first"
    metric_list:
      - metric: exact_match
        ignore_case: true
```

Pipelines without a `metric_list` inherit the task-level `metric_list`.

## Filter step execution

Filter steps execute in order, each receiving the output of the previous step. The contract:

- **Input**: `list[list[response]]` — a list of response lists, one per document
- **Output**: Same shape — the filter transforms responses but maintains the per-document grouping

The final step must reduce each document's response list to a single response (typically via `take_first`), which is then passed to the scorer.

## Adding a custom filter

Register a custom filter with the `@register_filter` decorator:

```python
from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("my_filter")
class MyFilter(Filter):
    def apply(self, resps, docs):
        # resps: list[list[str]] — responses grouped by document
        # docs: list[dict] — corresponding documents
        # Return: list[list[str]] — filtered responses
        return [[r.strip() for r in doc_resps] for doc_resps in resps]
```

Then reference it in YAML:

```yaml
filter_list:
  - name: "my-pipeline"
    filter:
      - function: "my_filter"
      - function: "take_first"
```

For more details on implementing custom filters, see [Custom Metrics & Filters](../extending/custom_metrics_and_filters.md).
