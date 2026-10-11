# ACLUE

### Paper

Can Large Language Model Comprehend Ancient Chinese? A Preliminary Test on ACLUE
https://arxiv.org/abs/2310.09550

The Ancient Chinese Language Understanding Evaluation (ACLUE) is an evaluation benchmark focused on ancient Chinese language comprehension. It aims to assess the performance of large-scale language models on understanding ancient Chinese. The benchmark comprises 15 tasks spanning various domains, including lexical, syntactic, semantic, inference, and knowledge. ACLUE's tasks are derived from a combination of manually curated questions from publicly available resources, and automatically
generated questions from classical Chinese language corpora. The range of questions span from the Xia dynasty (2070 BCE) to the Ming dynasty (1368 CE). ACLUE adopts a multiple-choice question format for all tasks.

Homepage: https://github.com/isen-zhang/ACLUE

### Dataset loading

These tasks use a local dataset helper to download and extract the original
ACLUE v1.0.0 archive, then read each subject's `test/{subject}.csv` and
`dev/{subject}.csv` with the built-in Hugging Face CSV loader and the original
six string fields. The archive URL is pinned to dataset revision
`f89544164a29a9eec66490bf6d0cd571ceab4358`. This avoids the dataset's remote
loading script, which is unsupported by Datasets 4 and later.

The subject names are retained as CSV configuration names. The test split,
development split and its ordered first-n few-shot examples, prompt format,
answer indices, and scoring metrics are unchanged.

### Changelog

- 2026-10-11: Version 1.0 -> 2.0. Read the original CSV files without a remote
  dataset script, preserving the evaluation inputs and scoring definitions.

### Citation

```bibtex
@inproceedings{zhang-li-2023-large,
    title = "Can Large Language Model Comprehend {A}ncient {C}hinese? A Preliminary Test on {ACLUE}",
    author = "Zhang, Yixuan  and Li, Haonan",
    booktitle = "Proceedings of the Ancient Language Processing Workshop",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.alp-1.9",
    pages = "80--87"
}
```

### Groups, Tags, and Tasks

#### Groups

- `aclue`: All 15 subjects of the ACLUE dataset, evaluated following the methodology in CMMLU's original implementation.

#### Tasks

The following tasks evaluate subjects in the ACLUE dataset using loglikelihood-based multiple-choice scoring:
- `aclue_{subject_english}`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
