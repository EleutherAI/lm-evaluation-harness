# Essay-Syntax

## Paper

Title: `Advancing Student Writing Through Automated Syntax Feedback`

Paper: https://arxiv.org/abs/2501.07740

Essay-Syntax-Instruct pairs English student essays with structured feedback that lists each
spelling, grammar or syntax error with its correction, the original sentence and an improved
sentence. The test set has 600 essays.

Homepage: https://github.com/KamyarZeinalipour/Essay-Syntax-Feedback

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/Essay-Syntax-Instruct

### Citation

```text
@inproceedings{zeinalipour2024advancing,
  title = {Advancing Student Writing Through Automated Syntax Feedback},
  author = {Kamyar Zeinalipour and Mehak Mehak and Fatemeh Parsamotamed and Marco Maggini and Marco Gori},
  booktitle = {International Workshop on AI in Education and Educational Research (AIEER 2024), at ECAI 2024},
  year = {2024},
  url = {https://arxiv.org/abs/2501.07740}
}
```

### Groups, Tags, and Tasks

#### Tasks

* `essay_syntax`: the model writes JSON feedback for an essay; corpus chrF against the reference feedback.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

This task reports chrF only. The paper's own evaluation scripts are in the GitHub repository;
numbers from the two setups are not directly comparable.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
