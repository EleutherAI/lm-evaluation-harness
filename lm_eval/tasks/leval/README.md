# L-Eval

### Paper

Title: `L-Eval: Instituting Standardized Evaluation for Long Context Language Models`

Abstract: `L-Eval is a long-context benchmark with 20 sub-tasks, 508 long documents, and
over 2,000 human-labeled query-response pairs covering diverse question styles, domains,
and input lengths (3k~200k tokens). It splits tasks into closed-ended groups (scored with
exact match) and open-ended groups (scored with n-gram metrics or an LLM judge).`

Homepage: `https://github.com/OpenLMLab/LEval`

Paper: `https://arxiv.org/abs/2307.11088`

### Citation

```
@inproceedings{an-etal-2024-l,
    title = "{L}-{E}val: Instituting Standardized Evaluation for Long Context Language Models",
    author = "An, Chenxin and Gong, Shansan and Zhong, Ming and Zhao, Xingjian and Li, Mukai and Zhang, Jun and Kong, Lingpeng and Qiu, Xipeng",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.776",
    pages = "14388--14411",
}
```

### Scope

This directory implements the **7 closed-ended (exam / exact-match) sub-tasks** of L-Eval.
These need no judge model, matching how LongBench is already integrated. The 13 open-ended
sub-tasks (scored by an LLM judge or n-gram metrics) are intentionally left for a follow-up
contribution.

The data is read straight from the benchmark's closed-ended `*.jsonl` files, pinned to a
fixed upstream commit (the original `L4NLP/LEval` Hugging Face dataset ships a loading
script, which `datasets >= 4` no longer executes). The task instructions are reproduced from
the benchmark so scores stay comparable with the paper; the answer extraction and
exact-match scoring in `utils.py` are an independent re-implementation of the behaviour of
L-Eval's `Evaluation/auto_eval.py` exam path (no code is copied from that GPL-3.0 project
into this MIT-licensed repository).

### Groups, Tags, and Tasks

#### Tags

* `leval`: All seven L-Eval closed-ended sub-tasks; run them together with `--tasks leval`.

#### Tasks

* `leval_coursera`: Multiple-choice questions (single **or** multiple correct options) over Coursera lecture transcripts.
* `leval_quality`: Single-answer multiple-choice reading comprehension (QuALITY).
* `leval_tpo`: Single-answer multiple-choice listening comprehension (TOEFL TPO).
* `leval_gsm100`: 100 harder grade-school math word problems answered with a long chain-of-thought prompt (GSM).
* `leval_codeu`: Predict the output of a function call after reading a long code base (CodeU).
* `leval_sci_fi`: `True`/`False` claims that must be judged against a science-fiction story.
* `leval_topic_retrieval`: Recover the 1st/2nd/3rd discussed topic from a long multi-topic conversation.

### Metric

Every sub-task reports `exact_match` in `[0, 1]`; the L-Eval paper reports the same quantity
multiplied by 100. Scoring follows the benchmark: exact set match for multiple choice (with
0.25 partial credit for coursera's multi-answer questions), numeric equality for `gsm100`,
and case-insensitive containment of the gold string for `codeU` / `sci_fi` / `topic_retrieval`.

### Notes / deviations

* `leval_sci_fi` scores only the in-document (loyalty) `True`/`False` verdict. L-Eval's
  optional second pass, which re-asks the model to contrast the story with real-world facts,
  is not reproduced because the harness generates a single completion per request.
* Documents are passed through as-is; L-Eval's original fixed-budget left/right truncation is
  left to the model wrapper's own context handling.
* `until` is empty, so generation stops at `max_gen_toks`. The extraction functions parse the
  answer out of the raw generation, matching the benchmark.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
