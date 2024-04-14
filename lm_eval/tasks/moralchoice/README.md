# MoralChoice

### Paper

Title: `Evaluating the Moral Beliefs Encoded in LLMs`

Abstract: https://arxiv.org/abs/2307.14324

This paper presents a case study on the design, administration, post-processing, and evaluation of surveys on large language models (LLMs). It comprises two components:

A statistical method for eliciting beliefs encoded in LLMs. We introduce statistical measures and evaluation metrics that quantify the probability of an LLM "making a choice", the associated uncertainty, and the consistency of that choice.

Homepage: https://github.com/PKU-Alignment/beavertails

### Citation

```
@inproceedings{scherrer2023evaluating,
  title={Evaluating the Moral Beliefs Encoded in LLMs},
  author={Nino Scherrer and Claudia Shi and Amir Feder and David Blei},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=O06z2G18me}
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
