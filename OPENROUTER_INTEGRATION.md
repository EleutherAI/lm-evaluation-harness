# OpenRouter Statistics Integration

This fork adds comprehensive OpenRouter statistics integration to lm-evaluation-harness, enabling per-sample cost and performance tracking.

## 🚀 Features

### Sample-Level Statistics
- **Per-request tracking**: Each sample includes detailed OpenRouter statistics
- **Cost analysis**: Track exact cost per query/response pair
- **Performance metrics**: Latency, generation time, token usage
- **Model metadata**: Provider, model version, finish reason

### Enhanced Sample Records
Each sample in the JSONL output now includes an `openrouter_stats` field:

```json
{
  "doc_id": 0,
  "doc": {"question": "...", "answer": "..."},
  "target": "18",
  "arguments": [...],
  "resps": [...],
  "filtered_resps": ["18"],
  "filter": "flexible-extract",
  "exact_match": 1,
  "openrouter_stats": {
    "total_cost": 0.00020945,
    "prompt_tokens": 118,
    "completion_tokens": 55,
    "generation_time": 7651,
    "latency": 4833,
    "model": "openai/gpt-5-nano-2025-08-07",
    "provider_name": "OpenAI",
    "finish_reason": "stop",
    "native_tokens_prompt": 117,
    "native_tokens_completion": 509
  }
}
```

## 🔧 Usage

### Basic Command
```bash
lm_eval --model openrouter-chat-completions \
        --model_args model=google/gemini-2.5-flash \
        --tasks gsm8k \
        --num_fewshot 0 \
        --log_samples \
        --output_path ./results/Gemini-2-5-Flash_gsm8k_results.json \
        --apply_chat_template \
        --gen_kwargs max_gen_toks=20000
```

### Supported Models
All OpenRouter models are supported, including:
- OpenAI: `openai/gpt-4o`, `openai/gpt-4o-mini`
- Google: `google/gemini-2.5-flash`, `google/gemini-1.5-pro`
- Anthropic: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`
- Meta: `meta-llama/llama-3.1-405b-instruct`
- And many more...

## 📊 Benefits

### Cost Analysis
- Track exact spending per benchmark
- Compare cost-effectiveness across models
- Identify expensive queries for optimization

### Performance Insights
- Analyze latency patterns
- Understand token efficiency
- Monitor generation speed

### Research Applications
- Cost-aware model comparison
- Performance profiling
- Resource optimization studies

## 🛠️ Technical Implementation

### Modified Files
1. **`lm_eval/api/instance.py`**: Added `stats` field to Instance class
2. **`lm_eval/models/openai_completions.py`**: Enhanced OpenRouter integration
3. **`lm_eval/evaluator.py`**: Updated sample logging with statistics

### Backward Compatibility
- Non-OpenRouter models work unchanged
- Existing workflows remain compatible
- Optional statistics don't break existing code

## 🎯 Use Cases

### Model Comparison
Compare different models on the same benchmark with cost and performance metrics.

### Budget Planning
Estimate costs for large-scale evaluations before running them.

### Performance Optimization
Identify bottlenecks and optimize for speed or cost.

### Research Analysis
Conduct comprehensive studies with detailed per-sample data.

## 📈 Example Analysis

With this integration, you can easily analyze:
- Which models provide best value for money
- How performance varies across different query types
- Cost distribution across benchmark tasks
- Latency patterns and outliers

## 🔄 Version Management

This fork maintains compatibility with the upstream lm-evaluation-harness while adding OpenRouter-specific enhancements. Regular merges from upstream ensure you get the latest features and bug fixes.
