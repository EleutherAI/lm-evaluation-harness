# ACPBench

### Paper

Title: ACPBench: Reasoning About Action, Change, and Planning
Abstract: https://arxiv.org/pdf/2410.05669

There is an increasing body of work using Large Language Models (LLMs) as agents for orchestrating workflows and making decisions in domains that require planning and multi-step reasoning. As a result, it is imperative to evaluate LMs on core skills required for planning. ACPBench is a benchmark for evaluating the reasoning tasks in the field of planning. The benchmark consists of 7 reasoning tasks over 13 planning domains. The collection is constructed from planning domains described in a formal language. This allows the synthesized problems to have provably correct solutions across many tasks and domains. Further, it allows the luxury to scale without additional human effort, i.e., many additional problems can be created automatically.

Homepage: https://ibm.github.io/ACPBench/


### Citation

```
@inproceedings{kokel2025acp
  author       = {Harsha Kokel and
                  Michael Katz and
                  Kavitha Srinivas and
                  Shirin Sohrabi},
  title        = {ACPBench: Reasoning about Action, Change, and Planning},
  booktitle    = {{AAAI}},
  publisher    = {{AAAI} Press},
  year         = {2025}
}
```

### Groups, Tags, and Tasks

#### Groups

* None

#### Tags

* `acp_bench` : Evaluates `acp_bool_cot_2shot` and `acp_mcq_cot_2shot`
* `acp_bool_cot_2shot` : Evaluates `acp_areach_bool`, `acp_app_bool`, `acp_just_bool`, `acp_land_bool`, `acp_prog_bool`, `acp_reach_bool`, `acp_val_bool` with chain-of-thought and 2 shots
* `acp_mcq_cot_2shot` : Evaluates `acp_areach_mcq`, `acp_app_mcq`, `acp_just_mcq`, `acp_land_mcq`, `acp_prog_mcq`, `acp_reach_mcq`, `acp_val_mcq`  with chain-of-thought and 2 shots

#### Tasks

7 Boolean tasks
* `acp_areach_bool`
* `acp_app_bool`
* `acp_just_bool`
* `acp_land_bool`
* `acp_prog_bool`
* `acp_reach_bool`
* `acp_val_bool`

7 MCQ tasks
* `acp_areach_mcq`
* `acp_app_mcq`
* `acp_just_mcq`
* `acp_land_mcq`
* `acp_prog_mcq`
* `acp_reach_mcq`
* `acp_val_mcq`

> ! The evaluation scripts are taken from original github https://github.com/IBM/ACPBench


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?


### Change Log

* 03/17/2025 Initial Commit
