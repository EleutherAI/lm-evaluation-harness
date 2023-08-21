# MC Taco

### Paper

Title: `"Going on a vacation" takes longer than "Going for a walk": A Study of Temporal Commonsense Understanding`
Abstract: https://arxiv.org/abs/1909.03065

MC-TACO is a dataset of 13k question-answer pairs that require temporal commonsense
comprehension. The dataset contains five temporal properties, (1) duration (how long
an event takes), (2) temporal ordering (typical order of events), (3) typical time
(when an event occurs), (4) frequency (how often an event occurs), and (5) stationarity
(whether a state is maintained for a very long time or indefinitely).

WARNING: Running this task with a `--limit` arg will give misleading results! The
corresponding dataset is structured such that each multiple-choice-question gathered
by the authors is split into question-option pairs, where each such pair gets
siloed into an individual document for plausibility testing. Because the harness
shuffles these documents, setting `--limit` will likely "cut off" certain candidate
answers. This is a problem because the task's metrics require an exhaustive evaluation
of a question's options. See section 4 of the paper for details.

Homepage: https://leaderboard.allenai.org/mctaco/submissions/public


### Citation

```
BibTeX-formatted citation goes here
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `mc_taco`


### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
