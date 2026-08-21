# NSINA

### Paper

Title: `NSINA: A News Corpus for Sinhala`

Abstract: https://arxiv.org/abs/2403.16571

NSINA is a collection of over 500,000 articles from popular Sinhala news websites, released alongside three benchmark tasks: news media identification, news category prediction, and news headline generation. Sinhala is a low-resource language spoken by over 20 million people, and NSINA provides the largest news corpus and benchmark suite available for it.

Homepage: https://github.com/Sinhala-NLP/NSINA

### Citation

```
@inproceedings{Nsina2024,
author={Hettiarachchi, Hansi and Premasiri, Damith and Uyangodage, Lasitha and Ranasinghe, Tharindu},
title={{NSINA: A News Corpus for Sinhala}},
booktitle={The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
year={2024},
month={May},
}
```

### Groups and Tasks

#### Groups

* `nsina`: Runs all NSINA tasks.

#### Tasks

* `nsina_categories`: 4-way news category classification (multiple choice, scored with accuracy and macro-F1). Prompts and answer choices are in Sinhala.
* `nsina_media`: 10-way news media source identification (multiple choice, scored with accuracy and macro-F1).
* `nsina_headlines`: headline generation from article content, scored with chrF. chrF is used instead of the ROUGE metrics reported in the paper because the default ROUGE tokenizer discards non-Latin-script text, which makes it unsuitable for Sinhala.

Note: the underlying datasets are gated on the Hugging Face Hub (automatic approval). You must be logged in with `hf auth login` and accept the terms on each dataset page before running these tasks.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The original paper evaluates fine-tuned encoder models (e.g. SinBERT); this implementation adapts the same data and splits for zero-/few-shot evaluation of autoregressive LMs, so scores are not directly comparable to the paper.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
