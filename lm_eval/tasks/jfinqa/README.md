# jfinqa

## Paper

Title: jfinqa: A Japanese Financial Numerical Reasoning QA Benchmark

Homepage: https://github.com/ajtgjmdjp/jfinqa

## Citation

```bibtex
@misc{ogawa2025jfinqa,
  title={jfinqa: A Japanese Financial Numerical Reasoning QA Benchmark},
  author={Ogawa, Saichi},
  year={2025},
  howpublished={\url{https://github.com/ajtgjmdjp/jfinqa}},
}
```

## Description

jfinqa is a benchmark of 927 questions for evaluating LLMs on numerical reasoning over Japanese corporate financial statements. Each question requires multi-step arithmetic (1-5 steps) over tables extracted from real EDINET filings, spanning 68 companies across J-GAAP, IFRS, and US-GAAP.

### Three subtasks:

| Subtask | Questions | Description |
|---------|-----------|-------------|
| Numerical Reasoning | 550 | Calculate financial metrics (growth rates, margins, ratios) |
| Consistency Checking | 200 | Verify internal consistency of reported figures |
| Temporal Reasoning | 177 | Determine direction of year-over-year changes |

### Baseline results (zero-shot, temperature=0, numerical matching with 1% tolerance):

| Model | Overall | NR | CC | TR |
|-------|---------|----|----|-----|
| GPT-4o | **84.9%** | 76.7% | **94.0%** | **100.0%** |
| GPT-4o-mini | 74.9% | **83.5%** | 88.0% | 33.3% |
| Gemini 2.0 Flash | 74.5% | 75.5% | 82.5% | 62.7% |

## Groups and Tasks

- **Group:** `jfinqa`
- **Tasks:**
  - `jfinqa_numerical` — Numerical reasoning (550 questions)
  - `jfinqa_consistency` — Consistency checking (200 questions)
  - `jfinqa_temporal` — Temporal reasoning (177 questions)

## Dataset

- HuggingFace: https://huggingface.co/datasets/ajtgjmdjp/jfinqa
- PyPI: https://pypi.org/project/jfinqa/
- License: Apache-2.0
- Source data: EDINET (Japan's Financial Services Agency)

## Checklist

- [x] Is the task an existing benchmark in the literature? — Yes, published with dataset, evaluation code, and baseline results.
- [x] Have you verified the samples from the dataset are correct? — Yes, all 927 questions have DSL-verified gold answers.
- [x] Is the dataset publicly available? — Yes, on HuggingFace Hub and PyPI.
- [x] Does the task have a dedicated README.md? — This file.
- [x] Have you cited the original paper? — Yes (see Citation above).
