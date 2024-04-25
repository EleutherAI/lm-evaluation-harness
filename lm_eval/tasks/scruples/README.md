# RACE

### Paper

Title: `SCRUPLES: A Corpus of Community Ethical Judgments on 32,000 Real-life Anecdotes`

Abstract: https://arxiv.org/abs/2008.09094

We introduce SCRUPLES, the first large-scale dataset with
625,000 ethical judgments over 32,000 real-life anecdotes.
Each anecdote recounts a complex ethical situation, often
posing moral dilemmas, paired with a distribution of judgments contributed by the community members. Our dataset
presents a major challenge to state-of-the-art neural language
models, leaving significant room for improvement. However, when presented with simplified moral situations, the results are considerably more promising, suggesting that neural
models can effectively learn simpler ethical building blocks.

Homepage: https://github.com/allenai/scruples

### Citation

```
@article{Lourie2020Scruples,
    author = {Nicholas Lourie and Ronan Le Bras and Yejin Choi},
    title = {Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes},
    journal = {arXiv e-prints},
    year = {2020},
    archivePrefix = {arXiv},
    eprint = {2008.09094},
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `scruples` : This task slightly change the original task from "WHO’S IN THE WRONG?" to "IS THE AUTHOR IN THE WRONG?" which is a binary classification task

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
