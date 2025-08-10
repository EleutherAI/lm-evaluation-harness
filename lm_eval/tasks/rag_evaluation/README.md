# RAG Evaluation Task

This task implements comprehensive RAG (Retrieval-Augmented Generation) evaluation using GPT-4 as a judge. It supports both individual metric evaluation and comprehensive multi-metric evaluation.

## Available Tasks

### Individual Metric Tasks

- **rag_evaluation_relevance**: Evaluates query relevance
- **rag_evaluation_completeness**: Evaluates query completeness
- **rag_evaluation_adherence**: Evaluates context adherence
- **rag_evaluation_context_completeness**: Evaluates context completeness
- **rag_evaluation_coherence**: Evaluates response coherence
- **rag_evaluation_length**: Evaluates response length
- **rag_evaluation_refusal_quality**: Evaluates refusal quality
- **rag_evaluation_refusal_clarification**: Evaluates refusal clarification quality
- **rag_evaluation_refusal_presence**: Evaluates refusal presence
- **rag_evaluation_correctness**: Evaluates correctness

### Comprehensive Task

- **rag_evaluation**: Evaluates all metrics in a single run

## Overview

The RAG evaluation task assesses the **query relevance** of model responses in RAG scenarios:

- **Query Relevance**: How relevant the response information is to the query
  - **Definition**: How much the information in the response is related to the query, and not necessarily how well it answers the query
  - **Range**: 0-10
  - **High Score**: All information in the response is relevant to the query
  - **Low Score**: Information in the response is largely irrelevant
  - **Requires Ground Truth**: No

## Task Configuration

- **rag_evaluation_relevance.yaml**: Evaluates query relevance of responses

## Data Format

Your dataset should be in JSON format with the following structure:

```json
[
  {
    "query": "What is the capital of France?",
    "context": [
      "Paris is the capital of France.",
      "France is a country in Europe."
    ],
    "ground_truth": "Paris is the capital of France.",
    "model_response": "Based on the context, Paris is the capital of France."
  }
]
```

## Usage

### Basic Usage

```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=your_model_name \
  --tasks rag_evaluation_relevance \
  --dataset_path your_data.json \
  --output_path results.json
```

### Using Example Data

```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=your_model_name \
  --tasks rag_evaluation_relevance \
  --output_path results.json
```

## Environment Setup

Make sure you have the OpenAI API key set:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Query Relevance Metric

### Definition

"How much the information in the response is related to the query, and not necessarily how well it answers the query."

### Scoring Criteria

- **High Score (8-10)**: All information in the response is relevant to the query
- **Medium Score (4-7)**: Most information is relevant, some may be tangential
- **Low Score (0-3)**: Information in the response is largely irrelevant
- **Score -1**: This metric is irrelevant for this response

### Examples

- **Good**: Query asks about "capital of France", response discusses Paris, France, and French geography
- **Poor**: Query asks about "capital of France", response discusses unrelated topics like weather or sports

## Customization

You can customize the evaluation by:

1. **Changing GPT model**: Modify the `gpt_model` parameter in `evaluate_with_gpt4` function
2. **Adjusting prompts**: Modify the prompt templates in `utils.py`
3. **Adding new metrics**: Extend `METRIC_DEFINITIONS` in `utils.py`

## Dependencies

- `openai`: For GPT-4 evaluation
- `torch`: For model inference
- `datasets`: For data processing
- `tqdm`: For progress bars

## Notes

- The evaluation uses GPT-4 as a judge, which requires an OpenAI API key
- Each evaluation call to GPT-4 incurs API costs
- Query relevance does not require ground truth answers
- The task focuses specifically on relevance, not correctness or completeness
