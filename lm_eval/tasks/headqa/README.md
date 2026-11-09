# HEAD-QA

### Paper

HEAD-QA: A Healthcare Dataset for Complex Reasoning
https://arxiv.org/pdf/1906.04701.pdf

HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to access a specialized position in the
Spanish healthcare system, and are challenging even for highly specialized humans. They are designed by the Ministerio
de Sanidad, Consumo y Bienestar Social.
The dataset contains questions about the following topics: medicine, nursing, psychology, chemistry, pharmacology and biology.

Homepage: https://aghie.github.io/head-qa/


### Citation

```
@inproceedings{vilares-gomez-rodriguez-2019-head,
    title = "{HEAD}-{QA}: A Healthcare Dataset for Complex Reasoning",
    author = "Vilares, David  and
      G{\'o}mez-Rodr{\'i}guez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1092",
    doi = "10.18653/v1/P19-1092",
    pages = "960--966",
    abstract = "We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans. We then consider monolingual (Spanish) and cross-lingual (to English) experiments with information retrieval and neural techniques. We show that: (i) HEAD-QA challenges current methods, and (ii) the results lag well behind human performance, demonstrating its usefulness as a benchmark for future work.",
}
```

### Groups and Tasks

#### Groups

- `headqa`: Evaluates `headqa_en` and `headqa_es`

#### Tasks

* `headqa_en` - English variant of HEAD-QA
* `headqa_es` - Spanish variant of HEAD-QA

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?\
  * [x] Same as LM Evaluation Harness v0.3.0 implementation
