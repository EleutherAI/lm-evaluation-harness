# WEBQs

### Paper

Title: `Semantic Parsing on Freebase from Question-Answer Pairs`

Abstract: `https://cs.stanford.edu/~pliang/papers/freebase-emnlp2013.pdf`

WebQuestions is a benchmark for question answering. The dataset consists of 6,642
question/answer pairs. The questions are supposed to be answerable by Freebase, a
large knowledge graph. The questions are mostly centered around a single named entity.
The questions are popular ones asked on the web (at least in 2013).

Homepage: `https://worksheets.codalab.org/worksheets/0xba659fe363cb46e7a505c5b6a774dc8a`


### Citation

```
@inproceedings{berant-etal-2013-semantic,
    title = "Semantic Parsing on {F}reebase from Question-Answer Pairs",
    author = "Berant, Jonathan  and
      Chou, Andrew  and
      Frostig, Roy  and
      Liang, Percy",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D13-1160",
    pages = "1533--1544",
}
```

### Groups and Tasks

#### Groups

* `freebase`

#### Tasks

* `webqs`: `Questions with multiple accepted answers.`

### Checklist

For adding novel benchmarks/datasets to the library:
  * [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
