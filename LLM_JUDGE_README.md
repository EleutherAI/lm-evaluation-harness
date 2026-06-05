# LLM-as-a-Judge Metric for lm-evaluation-harness

This implementation adds a flexible, configurable LLM-as-a-Judge metric to the lm-evaluation-harness framework. It allows you to evaluate model responses using a remote LLM via any OpenAI-compatible API.

## Features

- **Fully configurable via YAML** - All parameters can be set in task configuration files
- **Jinja2 prompt templates** - Full access to document fields via `{{ doc.field_name }}`, `{{ prediction }}`, `{{ reference }}`
- **Concurrent API calls** - Configurable concurrency (default: 32) for fast batch evaluation
- **Progress tracking** - tqdm progress bar during evaluation
- **Detailed logging** - Save prompts, responses, scores, and explanations to JSONL files
- **OpenAI-compatible** - Works with OpenAI, Claude (via LiteLLM), vLLM, Ollama, or any compatible endpoint
- **Automatic aggregation** - Built-in mean aggregation across all instances
- **Error handling** - Graceful fallback on API errors

## Files Modified

### Core Implementation
- **[lm_eval/api/metrics.py](lm_eval/api/metrics.py)**
  - `llm_judge_fn()` - Passthrough metric function
  - `llm_judge_agg()` - Aggregation function with concurrent API calls
  - `_call_llm_judge_single()` - Single API call helper
  - `_render_llm_judge_prompt()` - Jinja2 template rendering
  - `get_pending_llm_judge_details()` - Retrieves collected results for saving

- **[lm_eval/api/task.py](lm_eval/api/task.py)**
  - Modified `process_results()` to pass `(reference, prediction, doc, config)` tuples for llm_judge

- **[lm_eval/loggers/evaluation_tracker.py](lm_eval/loggers/evaluation_tracker.py)**
  - Added `save_llm_judge_details()` method for saving detailed results

- **[lm_eval/__main__.py](lm_eval/__main__.py)**
  - Added code to save LLM judge details after evaluation

## Quick Start

### 1. Install Dependencies

```bash
pip install openai jinja2 tqdm
```

### 2. Set API Key

```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Add to Task YAML

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    prompt_template: |
      You are an expert evaluator. Evaluate this response:

      Question: {{ doc.question }}
      Reference Answer: {{ reference }}
      Model Response: {{ prediction }}

      Score from 0-10 (start with "Score: X.XX"):
    model: gpt-4
    temperature: 0.0
    concurrency: 32
    save_details: true
```

### 4. Run Evaluation

LLM judge metrics are **disabled by default**. Use `--run_llm_judge` to enable them:

```bash
lm-eval run --model your_model \
            --tasks your_task \
            --output_path ./results \
            --run_llm_judge
```

## How It Works

### Architecture (Passthrough Pattern)

The LLM judge uses a passthrough/aggregation pattern similar to BLEU:

1. **`llm_judge_fn()`** - Passthrough function that collects `(reference, prediction, doc, config)` tuples
2. **`llm_judge_agg()`** - Aggregation function that:
   - Processes all items concurrently with ThreadPoolExecutor
   - Calls the LLM judge API for each item
   - Shows progress with tqdm
   - Stores detailed results for later saving
   - Returns the mean score

### Data Flow

```
YAML Config → Task → process_results() → llm_judge_fn() [passthrough]
                                               ↓
                                    Collect all (ref, pred, doc, config) tuples
                                               ↓
                                    llm_judge_agg() [concurrent API calls]
                                               ↓
                                    ThreadPoolExecutor + tqdm progress
                                               ↓
                                    Store results → EvaluationTracker saves JSONL
                                               ↓
                                    Return mean score
```

### Output Files

When `save_details: true` (default) and `--output_path` is specified, detailed results are saved to:

```
output_path/
  model_name_sanitized/
    results_<timestamp>.json
    samples_<task>_<timestamp>.jsonl
    llm_judge_<task>_<judge_model>_<timestamp>.jsonl  # LLM judge details
```

Each line in the JSONL file contains:
```json
{
  "idx": 0,
  "score": 8.5,
  "judgment_raw": "Score: 8.5\n\nThe translation is accurate...",
  "explanation": "The translation is accurate...",
  "formatted_prompt": "You are an expert...",
  "prediction": "Model's response",
  "reference": "Reference answer",
  "error": null
}
```

## Configuration Options

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `metric` | Must be `"llm_judge"` |
| `aggregation` | Must be `"llm_judge"` |
| `higher_is_better` | Set to `true` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | None | Unique name for this judge variant (for multiple judges) |
| `prompt_template` | str | Required | Jinja2 template for the judge prompt |
| `api_base` | str | env var | Base URL for API endpoint |
| `model` | str | env var or `"gpt-4"` | Model name for judging |
| `temperature` | float | `0.0` | Sampling temperature |
| `max_tokens` | int | `1024` | Max tokens in response |
| `concurrency` | int | `32` | Number of concurrent API calls |
| `save_details` | bool | `true` | Save detailed results to JSONL |
| `retry_attempts` | int | `3` | Retry attempts for transient errors |
| `retry_min_wait` | float | `1.0` | Min wait between retries (seconds) |
| `retry_max_wait` | float | `60.0` | Max wait between retries (seconds) |
| `max_error_rate` | float | `0.1` | Max error rate before failing (0.1 = 10%) |
| `preflight_check` | bool | `true` | Test API before batch evaluation |

**Note:** `api_key` is only supported via environment variables to prevent accidental exposure.

**Note:** Retry requires `tenacity` package: `pip install tenacity`

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_JUDGE_API_BASE` | Default API base URL |
| `LLM_JUDGE_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `LLM_JUDGE_MODEL` | Default model name |

For named judges (e.g., `name: accuracy`), use name-specific env vars:
- `LLM_JUDGE_ACCURACY_API_BASE`
- `LLM_JUDGE_ACCURACY_API_KEY`
- `LLM_JUDGE_ACCURACY_MODEL`

## Multiple LLM Judges

Use the `name` field to run multiple judges with different configs:

```yaml
metric_list:
  - metric: llm_judge
    name: accuracy
    prompt_template: "Evaluate factual accuracy... Score 0-10:"
    model: gpt-4
  - metric: llm_judge
    name: fluency
    prompt_template: "Evaluate fluency... Score 0-10:"
    model: claude-haiku-4-5-20251001
```

Results appear as separate metrics: `llm_judge_accuracy`, `llm_judge_fluency`

## Jinja2 Template Variables

The prompt template has access to:

| Variable | Description |
|----------|-------------|
| `{{ prediction }}` | The model's generated response |
| `{{ reference }}` | The reference/gold answer (if available) |
| `{{ doc }}` | The full document object with all fields |
| `{{ doc.field_name }}` | Any field from the dataset document |
| `{{ custom_var }}` | Any custom field from metric config or `extra_llm_judge_fields` |

### Dynamic Field Access

You can define custom variables in `metadata.extra_llm_judge_fields` and use them for dynamic field access. This is useful for base configs included by multiple task-specific configs:

```yaml
# base.yaml - define default variables
metadata:
  extra_llm_judge_fields:
    source_field: sentence_eng_Latn

metric_list:
  - metric: llm_judge
    prompt_template: |
      Source: {{ doc[source_field] }}
      Translation: {{ prediction }}
```

```yaml
# child.yaml - override via metadata
include: ../base.yaml
metadata:
  extra_llm_judge_fields:
    source_field: sentence_ara_Arab  # Override the source field
```

Variables from `extra_llm_judge_fields` are merged into the template context. You can also define variables directly in the metric config.

### Example: Translation Evaluation

```yaml
prompt_template: |
  You are an expert translation quality evaluator.

  Source Text ({{ doc.source_lang }}):
  {{ doc.source_text }}

  Reference Translation ({{ doc.target_lang }}):
  {{ reference }}

  Model Translation:
  {{ prediction }}

  Evaluate accuracy, fluency, and completeness.
  Score from 0-10 (start with "Score: X.XX"):
```

### Example: QA Evaluation

```yaml
prompt_template: |
  Question: {{ doc.question }}
  Context: {{ doc.context }}

  Reference Answer: {{ reference }}
  Model Answer: {{ prediction }}

  Is the model answer correct? Score 0-10:
```

## Usage Examples

### Example 1: Translation with Claude (via LiteLLM)

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    prompt_template: |
      Evaluate this translation from {{ doc.source_lang }} to {{ doc.target_lang }}:

      Source: {{ doc.source_text }}
      Translation: {{ prediction }}

      Score from 0-10 (start with "Score: X.XX"):
    model: claude-haiku-4-5-20251001
    api_base: https://your-litellm-endpoint.com
    temperature: 0.0
    concurrency: 32
```

### Example 2: Local vLLM Endpoint

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    api_base: http://localhost:8000/v1
    model: meta-llama/Llama-3-70B-Instruct
    api_key: dummy
    concurrency: 16
    prompt_template: |
      Rate this response 0-10:
      {{ prediction }}

      Score:
```

### Example 3: Disable Detail Logging

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    save_details: false  # Don't save JSONL details
    prompt_template: |
      ...
```

## Response Parsing

The default parser expects the judge response to start with:
```
Score: X.XX
```

Where X.XX is a numeric score. The parser:
- Extracts the score from the first line matching `Score: X.XX`
- Captures everything after the score line as the explanation
- Returns NaN if no valid score is found (no clamping)

## Structured JSON Outputs

For more reliable score extraction, use OpenAI's structured outputs by specifying `response_format` with example values:

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    response_format:
      score: 8.5
      accuracy: 9.0
      fluency: 8.0
      reasoning: "Good translation with minor issues."
    score_field: score  # Field to extract score from (default: "score")
    prompt_template: |
      Evaluate this translation:
      Source: {{ doc.source_text }}
      Translation: {{ prediction }}

      {{ response_format }}
    model: gpt-4o
```

### How It Works

1. JSON schema is inferred from the example values using `genson` (install: `pip install genson`)
2. OpenAI's structured outputs API ensures responses match the schema
3. Score is extracted from the specified `score_field`
4. Full parsed JSON is saved in `judgment_parsed` in the output

### Template Placeholder

Use `{{ response_format }}` in your prompt to include the formatted JSON example. If not present, instructions are auto-appended.

### Output Format

```json
{
  "idx": 0,
  "score": 8.5,
  "judgment_raw": "{\"score\": 8.5, \"accuracy\": 9.0, ...}",
  "judgment_parsed": {"score": 8.5, "accuracy": 9.0, "fluency": 8.0, "reasoning": "..."},
  "formatted_prompt": "...",
  "error": null
}
```

**Note:** Structured outputs require OpenAI models with JSON schema support (GPT-4o, GPT-4o-mini). May not work with all OpenAI-compatible endpoints.

## Important Considerations

### Cost

LLM-as-a-Judge can be expensive for large evaluations:
- **GPT-4**: ~$0.01-0.03 per evaluation
- **Claude Haiku**: ~$0.001-0.002 per evaluation
- **Self-hosted**: Free but requires resources

**Recommendations:**
- Test on small subsets first (`--limit 10`)
- Use cheaper models (Haiku, GPT-3.5) for development
- Use high concurrency to reduce wall-clock time

### Concurrency

The `concurrency` parameter controls parallel API calls:
- Higher values = faster evaluation but more API load
- Default of 32 works well for most endpoints
- Reduce if hitting rate limits

### Determinism

- Set `temperature: 0.0` for reproducibility
- Different models may still vary slightly
- Consider averaging multiple judge runs for critical evaluations

## Troubleshooting

### ImportError: openai not found
```bash
pip install openai
```

### ImportError: jinja2 not found
```bash
pip install jinja2
```

### API Authentication Failed
```bash
export OPENAI_API_KEY=your_key_here
```

### All scores are NaN
- Check API is responding (view logs with `LOGLEVEL=DEBUG`)
- Verify the judge response format includes "Score: X.XX"
- Check `llm_judge_*.jsonl` for raw responses

### Rate limit errors
- Reduce `concurrency` parameter
- Use a self-hosted endpoint for unlimited throughput

### JSONL files not being saved
- Ensure `--output_path` is specified
- Check that `save_details: true` (default)
- Verify write permissions to output directory

## License

Same as lm-evaluation-harness (MIT License)
