# ASDiv

### Paper

Title: `ASDiv: A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers`

Abstract: https://arxiv.org/abs/2106.15772

ASDiv (Academia Sinica Diverse MWP Dataset) is a diverse (in terms of both language
patterns and problem types) English math word problem (MWP) corpus for evaluating
the capability of various MWP solvers. Existing MWP corpora for studying AI progress
remain limited either in language usage patterns or in problem types. We thus present
a new English MWP corpus with 2,305 MWPs that cover more text patterns and most problem
types taught in elementary school. Each MWP is annotated with its problem type and grade
level (for indicating the level of difficulty).

NOTE: We currently ignore formulas for answer generation.

Homepage: https://github.com/chaochun/nlu-asdiv-dataset


### Citation

```
@misc{miao2021diverse,
    title={A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
    author={Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
    year={2021},
    eprint={2106.15772},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `asdiv`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
