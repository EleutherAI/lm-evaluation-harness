# WinoGrande

### Paper

Title: `WinoGrande: An Adversarial Winograd Schema Challenge at Scale`

Abstract: https://arxiv.org/abs/1907.10641

WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge
(Levesque, Davis, and Morgenstern 2011), but adjusted to improve the scale and
robustness against the dataset-specific bias. Formulated as a fill-in-a-blank
task with binary options, the goal is to choose the right option for a given
sentence which requires commonsense reasoning.

NOTE: This evaluation of Winogrande uses partial evaluation as described by
Trinh & Le in Simple Method for Commonsense Reasoning (2018).
See: https://arxiv.org/abs/1806.02847

Homepage: https://leaderboard.allenai.org/winogrande/submissions/public


### Citation

```
@article{sakaguchi2019winogrande,
    title={WinoGrande: An Adversarial Winograd Schema Challenge at Scale},
    author={Sakaguchi, Keisuke and Bras, Ronan Le and Bhagavatula, Chandra and Choi, Yejin},
    journal={arXiv preprint arXiv:1907.10641},
    year={2019}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `winogrande`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
