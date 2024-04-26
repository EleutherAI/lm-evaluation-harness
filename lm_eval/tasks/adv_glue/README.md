# Adversarial GLUE
**NOTE**: GLUE benchmark tasks do not provide publicly accessible labels for their test sets, so we default to the validation sets for all sub-tasks.

### Paper

Title: `Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models`

Abstract: https://openreview.net/forum?id=GF9cSKI3A_q

Adversarial GLUE Benchmark (AdvGLUE) is a comprehensive robustness evaluation benchmark that focuses on the adversarial robustness evaluation of language models. It covers five natural language understanding tasks from the famous GLUE tasks and is an adversarial version of GLUE benchmark.

Homepage: https://adversarialglue.github.io

### Citation

```
@article{Wang2021AdversarialGA,
  title={Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models},
  author={Boxin Wang and Chejian Xu and Shuohang Wang and Zhe Gan and Yu Cheng and Jianfeng Gao and Ahmed Hassan Awadallah and B. Li},
  journal={ArXiv},
  year={2021},
  volume={abs/2111.02840}
}
```

### Groups and Tasks

#### Groups

* `adv_glue`: Run all Glue subtasks.

#### Tasks

* `mnli`
* `mnli_mismatched`
* `mrpc`
* `qnli`
* `qqp`
* `rte`
* `sst`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
