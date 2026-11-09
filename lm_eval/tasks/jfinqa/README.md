# jfinqa

## Paper

Title: jfinqa: Japanese Financial Numerical Reasoning QA Benchmark

Homepage: https://github.com/ajtgjmdjp/jfinqa

## Citation

```bibtex
@misc{ogawa2025jfinqa,
  title={jfinqa: Japanese Financial Numerical Reasoning QA Benchmark},
  author={Ogawa, Saichi},
  year={2025},
  url={https://github.com/ajtgjmdjp/jfinqa},
}
```

## Description

jfinqa is a benchmark of 1000 questions for evaluating LLMs on numerical reasoning over Japanese corporate financial statements. Each question requires multi-step arithmetic (1-5 steps) over tables extracted from real EDINET filings, spanning 68 companies across J-GAAP, IFRS, and US-GAAP.

### Three subtasks:

| Subtask | Questions | Description |
|---------|-----------|-------------|
| Numerical Reasoning | 550 | Calculate financial metrics (growth rates, margins, ratios) |
| Consistency Checking | 200 | Verify internal consistency of reported figures |
| Temporal Reasoning | 250 | Determine direction of year-over-year changes |

### Baseline results

| Model | Overall | Numerical Reasoning | Consistency Checking | Temporal Reasoning |
|-------|---------|--------------------|--------------------|-------------------|
| GPT-4o | **86.8%** | 79.6% | **93.5%** | **97.2%** |
| Gemini 2.0 Flash | 77.3% | 78.7% | 82.5% | 70.0% |
| GPT-4o-mini | 69.0% | **83.6%** | 86.0% | 23.2% |

*1000 questions, zero-shot, temperature=0, numerical matching with 1% tolerance*

## Groups and Tasks

- **Group:** `jfinqa`
- **Tasks:**
  - `jfinqa_numerical` — Numerical reasoning (550 questions)
  - `jfinqa_consistency` — Consistency checking (200 questions)
  - `jfinqa_temporal` — Temporal reasoning (250 questions)

## Dataset

- HuggingFace: https://huggingface.co/datasets/ajtgjmdjp/jfinqa
- PyPI: https://pypi.org/project/jfinqa/
- License: Apache-2.0
- Source data: EDINET (Japan's Financial Services Agency)

## Checklist

- [x] Is the task an existing benchmark in the literature? — Yes, published with dataset, evaluation code, and baseline results.
- [x] Have you verified the samples from the dataset are correct? — Yes, all 1000 questions have DSL-verified gold answers.
- [x] Is the dataset publicly available? — Yes, on HuggingFace Hub and PyPI.
- [x] Does the task have a dedicated README.md? — This file.
- [x] Have you cited the original paper? — Yes (see Citation above).
