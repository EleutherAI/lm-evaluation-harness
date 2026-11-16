# ANLI

### Paper

Title: `Adversarial NLI: A New Benchmark for Natural Language Understanding`

Paper Link: https://arxiv.org/abs/1910.14599

Adversarial NLI (ANLI) is a dataset collected via an iterative, adversarial
human-and-model-in-the-loop procedure. It consists of three rounds that progressively
increase in difficulty and complexity, and each question-answer includes annotator-
provided explanations.

Homepage: https://github.com/facebookresearch/anli

### Citation

```
@inproceedings{nie-etal-2020-adversarial,
    title = "Adversarial {NLI}: A New Benchmark for Natural Language Understanding",
    author = "Nie, Yixin  and
      Williams, Adina  and
      Dinan, Emily  and
      Bansal, Mohit  and
      Weston, Jason  and
      Kiela, Douwe",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```

### Groups and Tasks

#### Groups

* `anli`: Evaluates `anli_r1`, `anli_r2`, and `anli_r3`

#### Tasks
* `anli_r1`: The data collected adversarially in the first round.
* `anli_r2`: The data collected adversarially in the second round, after training on the previous round's data.
* `anli_r3`: The data collected adversarially in the third round, after training on the previous multiple rounds of data.


### Checklist

For adding novel benchmarks/datasets to the library:
  * [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
