# QASPER

### Paper

Title: `A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers`

Abstract: https://arxiv.org/abs/2105.03011

QASPER is a dataset of 5,049 questions over 1,585 Natural Language Processing papers.
Each question is written by an NLP practitioner who read only the title and abstract
of the corresponding paper, and the question seeks information present in the full
text. The questions are then answered by a separate set of NLP practitioners who also
provide supporting evidence to answers.

Homepage: https://allenai.org/data/qasper

### Citation

```
@article{DBLP:journals/corr/abs-2105-03011,
    author    = {Pradeep Dasigi and
               Kyle Lo and
               Iz Beltagy and
               Arman Cohan and
               Noah A. Smith and
               Matt Gardner},
    title     = {A Dataset of Information-Seeking Questions and Answers Anchored in
               Research Papers},
    journal   = {CoRR},
    volume    = {abs/2105.03011},
    year      = {2021},
    url       = {https://arxiv.org/abs/2105.03011},
    eprinttype = {arXiv},
    eprint    = {2105.03011},
    timestamp = {Fri, 14 May 2021 12:13:30 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2105-03011.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Groups and Tasks

#### Groups

* `qasper`: executes both `qasper_bool` and `qasper_freeform`

#### Tasks

* `qasper_bool`: Multiple choice task that evaluates the task with `answer_type="bool"`
* `qasper_freeform`: Greedy generation task that evaluates the samples from the task with `answer_type="free form answer"`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
