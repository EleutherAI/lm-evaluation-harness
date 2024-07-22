# TruthfulQA

### Paper

Title: `TruthfulQA: Measuring How Models Mimic Human Falsehoods`
Abstract: `https://arxiv.org/abs/2109.07958`

Homepage: `https://github.com/sylinrl/TruthfulQA`


### Citation

```
@inproceedings{lin-etal-2022-truthfulqa,
    title = "{T}ruthful{QA}: Measuring How Models Mimic Human Falsehoods",
    author = "Lin, Stephanie  and
      Hilton, Jacob  and
      Evans, Owain",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.229",
    doi = "10.18653/v1/2022.acl-long.229",
    pages = "3214--3252",
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `truthfulqa_mc1`: `Multiple-choice, single answer`
* `truthfulqa_mc2`: `Multiple-choice, multiple answers`
* `truthfulqa_gen`: `Answer generation`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
