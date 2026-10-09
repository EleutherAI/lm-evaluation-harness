# Saraiki LLM Benchmark

### Paper

No paper yet; this benchmark is introduced here and on its dataset page.

A 7-task benchmark for evaluating language models in Saraiki (Jataki/Multani, Shahmukhi script).
Items are adapted from original English benchmarks (MMLU, Belebele, HellaSwag, GSM8K, IFBench/IFEval,
TruthfulQA, MMLU-Pro), plus a set of safety prompts.

Homepage: https://huggingface.co/datasets/themohal/saraiki-llm-bench

The dataset is gated: accept the access conditions on the dataset page and log in with
`hf auth login` before running. `saraiki_instruction` also needs
`pip install "git+https://github.com/allenai/IFBench.git"`.

### Citation

```bibtex
@misc{raza2026saraikillmbench,
  title        = {Saraiki LLM Benchmark},
  author       = {Raza, Muhammad Farjad Ali},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/themohal/saraiki-llm-bench}}
}
```

### Groups, Tags, and Tasks

#### Groups

* `saraiki_bench`: all eight tasks below

#### Tasks

* `saraiki_knowledge`: MMLU, culturally neutral subjects (5-shot, acc)
* `saraiki_knowledge_hard`: MMLU-Pro, neutral categories, up to 10 options (5-shot, acc)
* `saraiki_commonsense`: HellaSwag (5-shot, acc)
* `saraiki_reading`: Belebele (5-shot, acc)
* `saraiki_math`: GSM8K (8-shot, exact_match on the last number)
* `saraiki_truthfulqa`: TruthfulQA mc1 (0-shot, acc)
* `saraiki_instruction`: IFBench + IFEval constraints (0-shot, prompt/instruction-level strict and loose acc)
* `saraiki_safety`: refusal rate on harmful requests (higher is better) and over-refusal rate on benign requests (lower is better); keyword-based refusal detection

### Checklist

For adding novel benchmarks/datasets to the library:

* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

New benchmark introduced with this PR (no paper yet). Each task follows the scoring of its source
benchmark (MMLU, Belebele, HellaSwag, GSM8K, IFBench/IFEval, TruthfulQA, MMLU-Pro); baseline results
are in the PR description.

### Changelog
