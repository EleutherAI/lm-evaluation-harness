# Task-name

### Paper

Title: `COMMONSENSEQA: A Question Answering Challenge Targeting
Commonsense Knowledge`

Abstract: https://arxiv.org/pdf/1811.00937.pdf

CommonsenseQA is a multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers.
It contains 12,102 questions with one correct answer and four distractor answers.

Homepage: https://www.tau-nlp.org/commonsenseqa


### Citation

```
@inproceedings{talmor-etal-2019-commonsenseqa,
    title = "{C}ommonsense{QA}: A Question Answering Challenge Targeting Commonsense Knowledge",
    author = "Talmor, Alon  and
      Herzig, Jonathan  and
      Lourie, Nicholas  and
      Berant, Jonathan",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1421",
    doi = "10.18653/v1/N19-1421",
    pages = "4149--4158",
    archivePrefix = "arXiv",
    eprint        = "1811.00937",
    primaryClass  = "cs",
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `commonsense_qa`: Represents the "random" split from the paper. Uses an MMLU-style prompt, as (presumably) used by Llama evaluations.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
