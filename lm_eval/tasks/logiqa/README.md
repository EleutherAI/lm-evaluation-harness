# LogiQA

### Paper
LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning

LogiQA is a dataset for testing human logical reasoning. It consists of 8,678 QA\ninstances, covering multiple types of deductive reasoning. Results show that state-of-the-art
neural models perform by far worse than human ceiling. The dataset can also serve as a benchmark for reinvestigating logical AI under the deep learning NLP setting.

Homepage: https://github.com/lgw863/LogiQA-dataset

### Citation

```bibtex
@misc{liuLogiQAChallengeDataset2020,
  title = {{{LogiQA}}: {{A Challenge Dataset}} for {{Machine Reading Comprehension}} with {{Logical Reasoning}}},
  shorttitle = {{{LogiQA}}},
  author = {Liu, Jian and Cui, Leyang and Liu, Hanmeng and Huang, Dandan and Wang, Yile and Zhang, Yue},
  date = {2020-07-16},
  number = {arXiv:2007.08124},
  eprint = {2007.08124},
  eprinttype = {arxiv},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2007.08124},
  url = {http://arxiv.org/abs/2007.08124},
}
```

[//]: # (### Subtasks)


### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] The original paper evaluates on unreleased BERT and RoBERTa models.


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  * [x] Same as LM Evaluation Harness v0.3.0 implementation
