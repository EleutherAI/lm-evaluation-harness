# SimpleQA

### Paper

**SimpleQA: Measuring Short-Form Factuality in Large Language Models**
OpenAI, October 2024
https://openai.com/index/introducing-simpleqa/

### Description

SimpleQA is a benchmark for evaluating the **short-form factual accuracy** of language models. It consists of 4,326 fact-seeking questions, each with exactly one unambiguous, verifiable answer that does not change over time.

Every question in SimpleQA was designed to:
- Have a **single indisputable correct answer** (easy to grade)
- **Not change over time** (avoids "who is the current president?" style questions)
- **Induce hallucinations** from GPT-4o or GPT-3.5 during dataset construction (ensures questions are non-trivial)

SimpleQA is fundamentally different from reasoning benchmarks like MATH or GPQA. It measures **parametric knowledge** — what facts did the model memorize during training? — rather than the ability to reason through novel problems.

### Metrics

| Metric | Description |
|---|---|
| `exact_match` | 1.0 if the normalized model output exactly matches the normalized reference answer |
| `f1` | Token-level F1 score (partial credit for partially correct answers) |
| `not_attempted` | Fraction of questions where the model explicitly declined to answer |

### Grading Details

Answers are normalized before comparison:
- Lowercased
- Punctuation removed
- Leading articles (a, an, the) dropped
- Whitespace collapsed

The original paper uses GPT-4o as a judge to grade answers as **correct / incorrect / not attempted**. This implementation uses deterministic string matching for offline evaluation — no API calls required. Users who want model-graded evaluation can follow the judge pattern used in the `gpqa` task.

### Note on `not_attempted`

Unlike most benchmarks that only track accuracy, SimpleQA tracks whether the model says "I don't know." A lower `not_attempted` rate is not always better — a model that confidently hallucinates has lower not_attempted but is arguably *worse* than one that hedges. The calibration between these three rates (correct, incorrect, not_attempted) tells a richer story about model reliability.

### Recommended Use

```bash
# Zero-shot (matches original paper setup)
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks simpleqa \
    --device cuda:0 \
    --batch_size 8

# Few-shot (exploratory)
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks simpleqa \
    --num_fewshot 5 \
    --device cuda:0
```

### Dataset

- **HuggingFace path**: `basicv8vc/SimpleQA`
- **Split used**: `train` (this is the full test set — 4,326 examples)
- **Fields**: `problem` (question), `answer` (ground truth), `metadata` (topic info)

### Citation

```bibtex
@misc{openai2024simpleqa,
  author       = {OpenAI},
  title        = {SimpleQA: Measuring Short-Form Factuality in Large Language Models},
  year         = {2024},
  howpublished = {\url{https://openai.com/index/introducing-simpleqa/}},
}
```
