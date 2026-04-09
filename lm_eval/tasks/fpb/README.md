# FPB

### Paper

Title: Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts

Abstract: https://arxiv.org/abs/1307.5336

Polar sentiment dataset of sentences from financial news. The dataset consists of 4840 sentences from English language financial news categorised by sentiment.
The eval uses 5-shot prompting to determing the sentiment of finance related sentences.
The subset sentences_50agree is used which contains sentences whose sentiment was agreed upon by at least 50 percent of reviewers.

### Citation

```
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
```

### Groups and Tasks

#### Tasks

* `fpb`: `5-shot sentiment analysis of sentences_50agree subset of financial phrasebank`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
