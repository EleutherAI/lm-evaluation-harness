# SWAG

### Paper

Title: `SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference`

Abstract: https://arxiv.org/pdf/1808.05326.pdf

SWAG (Situations With Adversarial Generations) is an adversarial dataset
that consists of 113k multiple choice questions about grounded situations. Each
question is a video caption from LSMDC or ActivityNet Captions, with four answer
choices about what might happen next in the scene. The correct answer is the
(real) video caption for the next event in the video; the three incorrect
answers are adversarially generated and human verified, so as to fool machines
but not humans.

Homepage: https://rowanzellers.com/swag/


### Citation

```
@inproceedings{zellers2018swagaf,
    title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
    author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year={2018}
}
```

### Groups and Tasks

#### Groups

* Not a part of a task yet.

#### Tasks

* `swag`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
