# HarmBench

## Paper

Title: HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

Abstract: https://arxiv.org/abs/2402.04249

Homepage: https://github.com/centerforaisafety/HarmBench

HarmBench is a standardized evaluation framework for automated red teaming.

### Citation

```text
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}```

### Implementation

The implementation follows the reference implementation provided in the HarmBench repository. We recommend using the `VLLM_WORKER_MULTIPROC_METHOD=spawn` environment variable when running the HarmBench task.

### Groups, Tags, and Tasks

#### Groups

* `harmbench`: `Short description`

#### Tasks

* `harmbench_direct_request`: `1-sentence description of what this particular task does`
* `harmbench_human_jailbreaks`: ...

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
