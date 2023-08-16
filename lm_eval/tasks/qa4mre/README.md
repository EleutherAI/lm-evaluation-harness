# QA4MRE

### Paper

Title: `QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation`

Abstract: https://www.cs.cmu.edu/~./hovy/papers/13CLEF-QA4MRE.pdf

The (English only) QA4MRE challenge which was run as a Lab at CLEF 2011-2013.
The main objective of this exercise is to develop a methodology for evaluating
Machine Reading systems through Question Answering and Reading Comprehension
Tests. Systems should be able to extract knowledge from large volumes of text
and use this knowledge to answer questions. Four different tasks have been
organized during these years: Main Task, Processing Modality and Negation for
Machine Reading, Machine Reading of Biomedical Texts about Alzheimer's disease,
and Entrance Exam.

Homepage: http://nlp.uned.es/clef-qa/repository/qa4mre.php


### Citation

```
@inproceedings{Peas2013QA4MRE2O,
    title={QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation},
    author={Anselmo Pe{\~n}as and Eduard H. Hovy and Pamela Forner and {\'A}lvaro Rodrigo and Richard F. E. Sutcliffe and Roser Morante},
    booktitle={CLEF},
    year={2013}
}
```

### Groups and Tasks

#### Groups

* `qa4mre`

#### Tasks

* `qa4mre_2011`
* `qa4mre_2012`
* `qa4mre_2013`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
