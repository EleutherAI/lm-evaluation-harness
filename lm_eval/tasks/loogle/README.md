# LooGLE

### Paper

Title: `LooGLE: Can Long-Context Language Models Understand Long Contexts?`

Abstract: `LooGLE is a benchmark for long-context understanding built from documents
published after 2022 (average ~20k words). It contains short-dependency tasks (short-
dependency QA and cloze) and long-dependency tasks (long-dependency QA and summarization),
where answering requires aggregating evidence spread across the document.`

Homepage: `https://github.com/bigai-nlco/LooGLE`

Paper: `https://arxiv.org/abs/2311.04939`

### Citation

```
@inproceedings{li-etal-2024-loogle,
    title = "{L}oo{GLE}: Can Long-Context Language Models Understand Long Contexts?",
    author = "Li, Jiaqi and Wang, Mengmeng and Zheng, Zilong and Zhang, Muhan",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.859",
    pages = "16304--16333",
}
```

### Scope

This directory implements the three **free-text generation** sub-tasks of LooGLE, scored
with ROUGE and needing no judge model (mirroring how LongBench is integrated). The
`shortdep_cloze` sub-task (entity fill-in, scored by exact/partial entity match) is
intentionally left for a follow-up.

The data is loaded from the Hugging Face hub (`bigai-nlco/LooGLE`, MIT-licensed). The
prompts are reproduced from the benchmark's `config/task2prompt.json`; ROUGE is
re-implemented in `utils.py` using only the standard library, so no extra dependency is
required.

### Groups, Tags, and Tasks

#### Tags

* `loogle`: The three LooGLE generation sub-tasks; run them together with `--tasks loogle`.

#### Tasks

* `loogle_shortdep_qa`: Short-dependency question answering — the answer relies on a single span in the document.
* `loogle_longdep_qa`: Long-dependency question answering — the answer requires aggregating evidence from multiple, distant parts of the document.
* `loogle_summarization`: Abstractive summarization of a long scientific paper.

### Metric

Each task reports `rouge_l` (primary), `rouge_1` and `rouge_2`, as F-measure in `[0, 1]`.
ROUGE is computed over lowercased, punctuation-stripped whitespace tokens (LCS for
ROUGE-L, n-gram overlap for ROUGE-1/2). The LooGLE paper additionally reports the recall
variant of ROUGE alongside BLEU, METEOR, BERTScore and a GPT-4 judge; only the
dependency-free n-gram metric is implemented here.

### Notes / deviations

* Documents are passed through as-is; truncation to the model's context window is left to
  the model wrapper.
* `until` is empty, so generation stops at `max_gen_toks` (300 for `shortdep_qa`, 500 for
  `longdep_qa` and `summarization`, matching `config/task2maxlen.json`).

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
