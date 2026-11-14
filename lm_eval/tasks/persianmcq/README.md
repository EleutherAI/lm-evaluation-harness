# PersianMCQ

## Paper

Title: `PersianMCQ-Instruct: A Comprehensive Resource for Generating Multiple-Choice Questions in Persian`

Paper: https://aclanthology.org/2025.loreslm-1.27/

PersianMCQ-Instruct contains over 12,000 Persian multiple-choice questions generated from Persian
Wikipedia passages and refined, released to train and evaluate Persian question generation.

Homepage: https://github.com/KamyarZeinalipour/Persian_MCQ

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/PersianMCQ-instruct

### Citation

```text
@inproceedings{zeinalipour-etal-2025-persianmcq,
    title = "{P}ersian{MCQ}-Instruct: A Comprehensive Resource for Generating Multiple-Choice Questions in {P}ersian",
    author = "Zeinalipour, Kamyar and Jamshidi, Neda and Akbari, Fahimeh and Maggini, Marco and Bianchini, Monica and Gori, Marco",
    booktitle = "Proceedings of the First Workshop on Language Models for Low-Resource Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.loreslm-1.27/",
    pages = "344--372"
}
```

### Groups, Tags, and Tasks

#### Tasks

* `persianmcq`: reading comprehension. The model reads the passage and a refined question; its log-likelihood of each of the four option texts picks the answer (accuracy). 11,271 test questions.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The paper uses the dataset for question generation. This task uses its refined questions, with
their marked correct options, as a question-answering test, which the paper does not report.
Options are scored by their text, without letters, so the order of the options does not matter.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
