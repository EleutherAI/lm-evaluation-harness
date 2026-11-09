# LogiQA

### Paper

Title: `LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning`

Abstract: https://arxiv.org/abs/2007.08124

LogiQA is a dataset for testing human logical reasoning. It consists of 8,678 QA
instances, covering multiple types of deductive reasoning. Results show that state-
of-the-art neural models perform by far worse than human ceiling. The dataset can
also serve as a benchmark for reinvestigating logical AI under the deep learning
NLP setting.

Homepage: https://github.com/lgw863/LogiQA-dataset


### Citation

```
@misc{liu2020logiqa,
    title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
    author={Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
    year={2020},
    eprint={2007.08124},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `logiqa`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
