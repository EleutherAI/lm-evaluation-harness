# LLMSQL

### Paper

Title: LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL
Abstract: https://arxiv.org/abs/2510.02350

LLMSQL is a cleaned, LLM-ready revision of WikiSQL. **LLMSQL 2.0** (the version used here) is a small, hard and verified
successor: 2,000 test questions over 942 Wikipedia tables. The questions cover three skills:

- **Lookups:** choosing the right filter column from terse WikiSQL-style questions.
- **Data conventions:** reading conventions in game logs, e.g. winner-first scores, where `L 37–0` means the team lost 0–37.
- **Text quantities:** comparing and subtracting dates stored as text.

Every reference answer was re-derived independently (by a second implementation, a hand audit, or both).

Homepage: https://github.com/LLMSQL/llmsql-benchmark
Dataset: https://huggingface.co/datasets/llmsql-bench/llmsql-2.0

### Evaluation

- **Prompt.** Zero-shot. The prompt is the dataset's `prompt` field: the table schema(s), the first 3 rows of every table
  shown and the question. The model is asked to return one SQLite query in a ```` ```sql ```` block.
- **Scoring.** The SQL is taken from the last ```` ```sql ```` block of the completion (falling back to the first
  `WITH`/`SELECT` statement). It is executed on the benchmark SQLite database, which is downloaded from the Hugging Face
  dataset on first use. The result is compared with the reference answer by a lenient execution match: insensitive to
  row order and duplicate rows, and tolerant to number formatting, units, currency signs and extra columns.
- **Metric.** `exec_acc`, the execution accuracy.

The scoring code is a copy of the reference implementation in the `llmsql` package. On the same gpt-oss-120b outputs it
gives the same per-question results (21.9% at medium reasoning effort). Running the reference SQL scores 100%.

Instruction-tuned and chat models should be run with `--apply_chat_template`. Reasoning models need a larger generation
budget, e.g. `--gen_kwargs max_gen_toks=16000`.

```bash
lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-Coder-7B-Instruct --tasks llmsql_2 --apply_chat_template
```

### Citation

```
@inproceedings{llmsql_bench,
  title={LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL},
  author={Pihulski, Dzmitry and Charchut, Karol and Novogrodskaia, Viktoria and Koco{\'n}, Jan},
  booktitle={2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  year={2025},
  organization={IEEE}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `llmsql_2`: LLMSQL 2.0, 2,000 test questions, zero-shot execution accuracy.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version (LLMSQL 2.0).
