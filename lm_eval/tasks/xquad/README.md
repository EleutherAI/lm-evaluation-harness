# XQuAD

### Paper

Title: `On the Cross-lingual Transferability of Monolingual Representations`

Abstract: https://aclanthology.org/2020.acl-main.421.pdf

XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. Consequently, the dataset is entirely parallel across 11 languages.

Homepage: https://github.com/deepmind/xquad


### Citation

```
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
```

### Groups and Tasks

#### Groups

* `xquad`: All available languages.

#### Tasks
Perform extractive question answering for each language's subset of XQuAD.
* `xquad_ar`: Arabic
* `xquad_de`: German
* `xquad_el`: Greek
* `xquad_en`: English
* `xquad_es`: Spanish
* `xquad_hi`: Hindi
* `xquad_ro`: Romanian
* `xquad_ru`: Russian
* `xquad_th`: Thai
* `xquad_tr`: Turkish
* `xquad_vi`: Vietnamese
* `xquad_zh`: Chinese



### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
