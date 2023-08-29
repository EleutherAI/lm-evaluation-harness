# ETHICS Dataset

### Paper

Pointer Sentinel Mixture Models
https://arxiv.org/pdf/1609.07843.pdf

The ETHICS dataset is a benchmark that spans concepts in justice, well-being,
duties, virtues, and commonsense morality. Models predict widespread moral
judgments about diverse text scenarios. This requires connecting physical and
social world knowledge to value judgements, a capability that may enable us
to steer chatbot outputs or eventually regularize open-ended reinforcement
learning agents.

Homepage: https://github.com/hendrycks/ethics

### Citation

```
@article{hendrycks2021ethics
    title={Aligning AI With Shared Human Values},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
```

### Groups and Tasks

#### Groups

- `hendrycks_ethics`

#### Tasks

* `ethics_cm`
* `ethics_deontology`
* `ethics_justice`
* `ethics_utilitarianism`
* (MISSING) `ethics_utilitarianism_original`
* `ethics_virtue`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
  * [ ] Matches v0.3.0 of Eval Harness
