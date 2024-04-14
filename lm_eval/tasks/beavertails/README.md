# BeaverTails

### Paper

Title: `BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset`

Abstract: https://arxiv.org/abs/2307.04657

BeaverTails is an AI safety-focused collection comprising a series of datasets. This repository includes human-labeled data consisting of question-answer (QA) pairs, each identified with their corresponding harm categories. It should be noted that a single QA pair can be associated with more than one category.

Homepage: https://github.com/PKU-Alignment/beavertails

### Citation

```
@article{beavertails,
  title   = {BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset},
  author  = {Jiaming Ji and Mickel Liu and Juntao Dai and Xuehai Pan and Chi Zhang and Ce Bian and Chi Zhang and Ruiyang Sun and Yizhou Wang and Yaodong Yang},
  journal = {arXiv preprint arXiv:2307.04657},
  year    = {2023}
}
```


### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
