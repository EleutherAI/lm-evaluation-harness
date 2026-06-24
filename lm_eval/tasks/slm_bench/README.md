# SLM-Bench

### Paper / Dataset

Title: `SLM-Bench: A Benchmark for Evaluating Sub-10M Language Models`

Homepage: https://huggingface.co/datasets/liodon-ai/slm-bench

### Citation

```bibtex
@dataset{slm_bench_2024,
  author = {Liodon AI},
  title = {SLM-Bench},
  year = {2024},
  url = {https://huggingface.co/datasets/liodon-ai/slm-bench},
  publisher = {Hugging Face}
}
```

### Description

SLM-Bench is a comprehensive benchmark designed specifically for evaluating sub-10M language models. It covers six core capability areas with 500 questions each (3,000 total), split into train/validation/test sets.

| Category | Description | Example |
|---|---|---|
| `arithmetic` | Math problems with plausible distractors | "What is 47 + 23?" |
| `pattern` | Sequence completion | "What comes next: 2, 6, 18, 54, ?" |
| `grammar` | Grammatical correctness MCQs | "She ___ to school every day." |
| `vocabulary` | WordNet-based vocab questions | Antonyms, synonyms, definitions |
| `logic` | Syllogism and logical reasoning | "Given: All A are B. Is C valid?" |
| `word_analogy` | Word relationship analogies | "hot : cold :: up : ?" |

### Groups, Tags, and Tasks

#### Groups

* `slm_bench`: Evaluates all six categories together (weighted aggregate)

#### Tags

* `slm_bench_tasks`: Includes all six subtasks

#### Tasks

* `slm_bench_arithmetic`: Math problems (500 questions)
* `slm_bench_pattern`: Sequence completion (500 questions)
* `slm_bench_grammar`: Grammar MCQs (500 questions)
* `slm_bench_vocabulary`: Vocabulary questions (500 questions)
* `slm_bench_logic`: Logical reasoning (500 questions)
* `slm_bench_word_analogy`: Word analogies (500 questions)

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
* [x] Referenced the original dataset
* [x] Followed lm-eval-harness task conventions

### Changelog

* Version 1.0: Initial addition of SLM-Bench task suite.
