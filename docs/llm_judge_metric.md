# LLM-as-a-Judge Metric

The `llm_judge` metric enables using a remote LLM (via OpenAI-compatible API) to evaluate model responses. This is useful for tasks where automated metrics like BLEU or exact match are insufficient, and human-like judgment is needed.

## Features

- **Configurable via YAML**: All parameters can be set in task configuration files
- **Jinja2 prompt templates**: Full access to document fields via `{{ doc.field }}`, `{{ prediction }}`, `{{ reference }}`
- **Concurrent API calls**: Configurable concurrency (default: 32) for fast batch evaluation
- **Progress tracking**: tqdm progress bar during LLM judge evaluation
- **Detailed logging**: Save prompts, responses, scores, and explanations to JSONL files
- **OpenAI-compatible**: Works with OpenAI API, Claude (via LiteLLM), vLLM, Ollama, or any compatible endpoint
- **Automatic aggregation**: Built-in mean aggregation across all instances
- **Error handling**: Graceful fallback on API errors (returns NaN, no clamping)
- **Automatic retry**: Exponential backoff for transient API errors (rate limits, timeouts)
- **Pre-flight check**: Verify API connectivity before batch evaluation
- **Failure threshold**: Configurable max error rate to catch API issues

## Installation

The `llm_judge` metric requires the following packages:

```bash
pip install openai jinja2 tqdm tenacity
```

Optional packages:
- `genson` - Required for structured JSON outputs: `pip install genson`

## Enabling LLM Judge

LLM judge metrics are **disabled by default** to prevent accidental API costs. Use the `--run_llm_judge` flag to enable them:

```bash
lm-eval run --model hf \
            --model_args pretrained=gpt2 \
            --tasks my_task \
            --run_llm_judge \
            --output_path ./results
```

## Basic Usage

### Template Variables

The prompt template uses Jinja2 syntax and has access to:

| Variable | Description |
|----------|-------------|
| `{{ prediction }}` | The model's generated response |
| `{{ reference }}` | The reference/gold answer (from `doc_to_target`) |
| `{{ doc }}` | The full document object with all dataset fields |
| `{{ doc.field_name }}` | Any field from the dataset document |

### 1. Reference-Based Evaluation

Evaluate predictions against reference answers:

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    prompt_template: |
      Evaluate this response:
      Question: {{ doc.question }}
      Reference: {{ reference }}
      Prediction: {{ prediction }}

      Score from 0-10 (start with "Score: X.XX"):
    model: gpt-4
    temperature: 0.0
    concurrency: 32
```

### 2. Translation Evaluation

Access source text directly from the document:

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    prompt_template: |
      You are an expert translation evaluator.

      Source Text (English):
      {{ doc.sentence_eng_Latn }}

      Model Translation:
      {{ prediction }}

      Evaluate accuracy, fluency, and completeness.
      Score from 0-10 (start with "Score: X.XX"):
    model: claude-haiku-4-5-20251001
    api_base: https://your-litellm-endpoint.com
    concurrency: 32
    save_details: true
```

### 3. Prediction-Only Evaluation

Evaluate text quality without reference:

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    prompt_template: |
      Rate the quality of this text from 0-10:
      {{ prediction }}

      Score:
    model: gpt-4
```

## Configuration Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `metric` | Must be `"llm_judge"` |
| `aggregation` | Must be `"llm_judge"` |
| `higher_is_better` | Set to `true` |

### Optional Parameters

#### Naming (for multiple judges)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | None | Unique name for this judge variant (e.g., "accuracy", "fluency"). Creates distinct metric keys in results. |

#### API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_base` | str | env var or OpenAI URL | Base URL for OpenAI-compatible API |
| `model` | str | env var or `"gpt-4"` | Model name for judging |

**Note:** `api_key` is only supported via environment variables (not in YAML) to prevent accidental exposure.

#### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | `0.0` | Sampling temperature (use 0.0 for determinism) |
| `max_tokens` | int | `1024` | Maximum tokens in judge response |

#### Execution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concurrency` | int | `32` | Number of concurrent API calls |
| `save_details` | bool | `true` | Save detailed results to JSONL |

#### Reliability Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retry_attempts` | int | `3` | Number of retry attempts for transient API errors |
| `retry_min_wait` | float | `1.0` | Minimum wait time (seconds) between retries |
| `retry_max_wait` | float | `60.0` | Maximum wait time (seconds) between retries |
| `max_error_rate` | float | `0.1` | Maximum allowed error rate (0.1 = 10%). Evaluation fails if exceeded |
| `preflight_check` | bool | `true` | Run a test API call before batch evaluation to verify connectivity |

#### Prompt Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_template` | str | Required | Jinja2 template for the judge prompt |

## Environment Variables

Environment variables provide defaults that can be overridden by YAML config (except `api_key` which is env-only):

| Variable | Description |
|----------|-------------|
| `LLM_JUDGE_API_BASE` | Default API base URL |
| `LLM_JUDGE_API_KEY` | API key (falls back to `OPENAI_API_KEY`) |
| `LLM_JUDGE_MODEL` | Default model name |

### Named Variants

For named judges (e.g., `name: accuracy`), you can set judge-specific env vars:

| Variable | Description |
|----------|-------------|
| `LLM_JUDGE_ACCURACY_API_BASE` | API base for "accuracy" judge |
| `LLM_JUDGE_ACCURACY_API_KEY` | API key for "accuracy" judge |
| `LLM_JUDGE_ACCURACY_MODEL` | Model for "accuracy" judge |

Named env vars take precedence, then fall back to non-named vars, then defaults.

## Multiple LLM Judges

You can run multiple LLM judges with different configurations by using the `name` field:

```yaml
metric_list:
  - metric: llm_judge
    name: accuracy
    prompt_template: |
      Evaluate the factual accuracy of this response...
      Score from 0-10:
    model: gpt-4
  - metric: llm_judge
    name: fluency
    prompt_template: |
      Evaluate the fluency and naturalness of this response...
      Score from 0-10:
    model: claude-haiku-4-5-20251001
```

Results will be reported as separate metrics:

```json
{
  "llm_judge_accuracy": 8.5,
  "llm_judge_fluency": 7.2
}
```

Output files will include the judge name:
- `llm_judge_taskname_accuracy_gpt-4_timestamp.jsonl`
- `llm_judge_taskname_fluency_claude-haiku_timestamp.jsonl`

## Output Files

When `save_details: true` (default) and `--output_path` is specified, detailed results are saved to:

```
output_path/
  model_name_sanitized/
    results_<timestamp>.json
    samples_<task>_<timestamp>.jsonl
    llm_judge_<task>_<judge_model>_<timestamp>.jsonl
```

Each line in the JSONL file contains:

```json
{
  "idx": 0,
  "score": 8.5,
  "judgment_raw": "Score: 8.5\n\nThe response is accurate...",
  "explanation": "The response is accurate...",
  "formatted_prompt": "You are an expert evaluator...",
  "prediction": "Model's generated response",
  "reference": "Reference answer",
  "error": null
}
```

## Prompt Template Guidelines

### Jinja2 Syntax

Use double curly braces for variables:
- `{{ prediction }}` - model output
- `{{ reference }}` - gold answer
- `{{ doc.field_name }}` - any document field

Jinja2 also supports conditionals and loops if needed:

```yaml
prompt_template: |
  {% if reference %}
  Reference: {{ reference }}
  {% endif %}
  Response: {{ prediction }}
```

### Dynamic Field Access with Task Metadata

You can define custom variables in `metadata.extra_llm_judge_fields` and use them in templates for dynamic field access. This is useful for base configs that are included by multiple task-specific configs:

```yaml
# flores200_base.yaml
metadata:
  extra_llm_judge_fields:
    source_field: sentence_eng_Latn  # Default source field

metric_list:
  - metric: llm_judge
    prompt_template: |
      Source Text:
      {{ doc[source_field] }}

      Model Translation:
      {{ prediction }}
```

```yaml
# flores200_ara_Arab-eng_Latn.yaml (Arabic to English)
include: ../flores200_base.yaml
dataset_name: ara_Arab-eng_Latn

metadata:
  extra_llm_judge_fields:
    source_field: sentence_ara_Arab  # Override the source field
```

Variables from `extra_llm_judge_fields` are merged into the template context (metric-level config takes precedence if there's a conflict). You can also define custom variables directly in the metric config for metric-specific settings.

### Best Practices

1. **Clear scoring instructions**: Specify the scale (0-10) and what scores mean
2. **Structured format**: Request "Score: X.XX" at the start for reliable parsing
3. **Evaluation criteria**: List specific aspects to evaluate
4. **Access raw data**: Use `{{ doc.field }}` to access any dataset field

### Example Template

```yaml
prompt_template: |
  You are an expert evaluator. Assess the translation quality.

  Source Text ({{ doc.source_lang }}):
  {{ doc.source_text }}

  Reference Translation:
  {{ reference }}

  Model Translation:
  {{ prediction }}

  Evaluation Criteria:
  1. Accuracy - Does it convey the same meaning?
  2. Fluency - Is it natural in the target language?
  3. Completeness - Is all information preserved?

  Scoring Guide:
  10 = Perfect translation
  8-9 = Minor issues only
  6-7 = Notable problems but understandable
  4-5 = Significant errors
  0-3 = Major errors or incomplete

  Start your response with "Score: X.XX"
  Then explain your reasoning.
```

## Using Custom Endpoints

### vLLM Server

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
      Rate this answer from 0-10:
      Q: {{ doc.question }}
      A: {{ prediction }}
      Score:
```

### LiteLLM (for Claude, etc.)

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    api_base: https://your-litellm-proxy.com
    model: claude-haiku-4-5-20251001
    concurrency: 32
    prompt_template: |
      Score this response (0-10):
      {{ prediction }}
      Score:
```

### Ollama

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    api_base: http://localhost:11434/v1
    model: llama3:70b
    api_key: dummy
    prompt_template: |
      Score this response (0-10):
      {{ prediction }}
      Score:
```

## Response Parsing

The default parser expects the judge response to start with:

```
Score: X.XX
```

Where X.XX is a numeric score (any range). The parser:
- Extracts the score from the first line matching `Score: X.XX`
- Captures everything after the score line as the explanation
- Returns NaN if no valid score is found (no clamping applied)

## Structured JSON Outputs

For more reliable score extraction and richer structured responses, you can use OpenAI's structured outputs feature by specifying a `response_format` with example values.

### Installation

Structured outputs require the `genson` package for JSON schema inference:

```bash
pip install genson
```

### Basic Usage

Define `response_format` with example values that represent the expected response structure:

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    response_format:
      score: 8.5
      explanation: "The translation accurately conveys the meaning."
      accuracy: 9.0
      fluency: 8.0
    score_field: score  # Which field contains the score (default: "score")
    prompt_template: |
      Evaluate this translation:
      Source: {{ doc.source_text }}
      Translation: {{ prediction }}

      Respond in JSON format:
      {{ response_format }}
    model: gpt-4o
```

### How It Works

1. **Schema Inference**: The `response_format` example is converted to a JSON schema using `genson`
2. **API Call**: OpenAI's structured outputs API (`response_format.type: "json_schema"`) ensures the response matches the schema
3. **Score Extraction**: The score is extracted from the field specified by `score_field` (default: `"score"`)
4. **Full Response**: The complete parsed JSON is saved in `judgment_parsed` in the output JSONL

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_format` | dict | None | Example response structure with sample values |
| `score_field` | str | `"score"` | Field name to extract the numeric score from |

### Template Placeholder

Use `{{ response_format }}` in your prompt template to include the formatted JSON example:

```yaml
prompt_template: |
  Evaluate the response quality.

  Response: {{ prediction }}

  Provide your evaluation in this JSON format:
  {{ response_format }}
```

If `{{ response_format }}` is not present in the template but `response_format` is defined, instructions will be automatically appended to the prompt.

### Example: Multi-Dimensional Evaluation

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    response_format:
      score: 7.5
      accuracy: 8.0
      fluency: 7.0
      completeness: 7.5
      reasoning: "The translation captures the main ideas but misses some nuance."
    score_field: score
    prompt_template: |
      You are an expert translation evaluator.

      Source: {{ doc.source_text }}
      Reference: {{ reference }}
      Translation: {{ prediction }}

      Evaluate accuracy (0-10), fluency (0-10), and completeness (0-10).
      Calculate an overall score as the weighted average.

      {{ response_format }}
    model: gpt-4o
```

### Output with Structured Responses

When using structured outputs, the JSONL output includes the full parsed response:

```json
{
  "idx": 0,
  "score": 7.5,
  "judgment_raw": "{\"score\": 7.5, \"accuracy\": 8.0, ...}",
  "judgment_parsed": {
    "score": 7.5,
    "accuracy": 8.0,
    "fluency": 7.0,
    "completeness": 7.5,
    "reasoning": "The translation captures the main ideas..."
  },
  "formatted_prompt": "You are an expert...",
  "prediction": "Model's translation",
  "reference": "Reference translation",
  "error": null
}
```

### Benefits of Structured Outputs

1. **Reliable Parsing**: No regex-based score extraction that can fail
2. **Richer Data**: Capture multiple evaluation dimensions in a single call
3. **Type Safety**: Schema enforcement ensures consistent response structure
4. **Post-hoc Analysis**: Full JSON responses enable detailed analysis

### Limitations

- Requires OpenAI API with structured outputs support (GPT-4o, GPT-4o-mini, etc.)
- May not work with all OpenAI-compatible endpoints (vLLM, Ollama)
- Requires `genson` package for schema inference

## Cost Considerations

LLM-as-a-Judge can be expensive for large evaluations:

| Model | Cost per evaluation |
|-------|---------------------|
| GPT-4 | ~$0.01-0.03 |
| GPT-3.5 | ~$0.001-0.002 |
| Claude Haiku | ~$0.001-0.002 |
| Self-hosted | Free (requires GPU) |

**Recommendations:**
1. Test with `--limit 10` first
2. Use cheaper models (Haiku, GPT-3.5) for development
3. Use high concurrency to reduce wall-clock time
4. Consider self-hosted judges for large-scale evaluation

## Concurrency Tuning

The `concurrency` parameter controls parallel API calls:

- **Default (32)**: Works well for most cloud APIs
- **Higher values**: Faster but may hit rate limits
- **Lower values**: Safer for rate-limited endpoints
- **Self-hosted**: Can often use higher values (64+)

## Reliability Features

The LLM judge includes several features to improve reliability for large-scale evaluations.

### Pre-flight API Check

Before starting batch evaluation, a test API call verifies:
- API endpoint is reachable
- API key is valid
- Model name is correct

This catches configuration errors early, before wasting time and money. Disable with `preflight_check: false` if needed.

### Automatic Retry with Exponential Backoff

Transient errors (rate limits, server errors, timeouts) are automatically retried with exponential backoff:

```yaml
metric_list:
  - metric: llm_judge
    retry_attempts: 3      # Number of retries (default: 3)
    retry_min_wait: 1.0    # Min wait between retries in seconds
    retry_max_wait: 60.0   # Max wait between retries in seconds
```

Retryable errors include:
- Rate limit errors (HTTP 429)
- Server errors (HTTP 500, 502, 503, 504)
- Connection timeouts

Non-retryable errors (authentication, bad request) fail immediately.

**Note:** Requires `tenacity` package: `pip install tenacity`

### Failure Threshold

The evaluation fails if too many API calls error out:

```yaml
metric_list:
  - metric: llm_judge
    max_error_rate: 0.1   # Fail if >10% of calls error (default: 0.1)
```

This prevents silently returning misleading scores when the API is having issues. Set to `1.0` to disable.

### Example: Robust Configuration

```yaml
metric_list:
  - metric: llm_judge
    aggregation: llm_judge
    higher_is_better: true
    # Reliability settings
    preflight_check: true
    retry_attempts: 5
    retry_min_wait: 2.0
    retry_max_wait: 120.0
    max_error_rate: 0.05   # Strict: fail if >5% errors
    # Standard settings
    model: gpt-4o
    concurrency: 32
    prompt_template: |
      ...
```

## Troubleshooting

### ImportError: openai package not found

```bash
pip install openai
```

### ImportError: jinja2 not found

```bash
pip install jinja2
```

### API Authentication Failed

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Or specify in YAML:

```yaml
api_key: your_key_here
```

### All scores are NaN

1. Check the judge API is responding (use `LOGLEVEL=DEBUG`)
2. Verify score parsing - ensure response contains "Score: X.XX"
3. Check `llm_judge_*.jsonl` for raw responses and errors

### Rate limit errors

Reduce the `concurrency` parameter:

```yaml
concurrency: 8  # Lower for rate-limited endpoints
```

### JSONL files not being saved

1. Ensure `--output_path` is specified on the command line
2. Check that `save_details: true` (this is the default)
3. Verify write permissions to the output directory

## Implementation Details

### Architecture

The LLM judge uses a passthrough/aggregation pattern (like BLEU):

1. **`llm_judge_fn()`** - Passthrough metric that collects items
2. **`llm_judge_agg()`** - Aggregation function that:
   - Runs concurrent API calls via ThreadPoolExecutor
   - Shows progress with tqdm
   - Stores results for deferred saving via EvaluationTracker
   - Returns mean score

### Files

- **[lm_eval/api/metrics.py](../lm_eval/api/metrics.py)** - Core implementation
- **[lm_eval/api/task.py](../lm_eval/api/task.py)** - Task integration
- **[lm_eval/loggers/evaluation_tracker.py](../lm_eval/loggers/evaluation_tracker.py)** - Detail saving
- **[lm_eval/__main__.py](../lm_eval/__main__.py)** - CLI integration

## Related Metrics

- `exact_match`: For strict string matching
- `bleu`: For translation quality (corpus-level)
- `chrf`: For character-level matching
- Custom `process_results`: For complex domain-specific evaluation
