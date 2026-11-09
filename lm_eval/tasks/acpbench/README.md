# ACPBench

**Homepage:** https://ibm.github.io/ACPBench/

### Papers

**Title:** ACPBench: Reasoning About Action, Change, and Planning
**Pdf:** https://arxiv.org/pdf/2410.05669
**Task:** `acp_bench`
**Abstract:**

There is an increasing body of work using Large Language Models (LLMs) as agents for orchestrating workflows and making decisions in domains that require planning and multi-step reasoning. As a result, it is imperative to evaluate LMs on core skills required for planning. ACPBench is a benchmark for evaluating the reasoning tasks in the field of planning. The benchmark consists of 7 reasoning tasks over 13 planning domains. The collection is constructed from planning domains described in a formal language. This allows the synthesized problems to have provably correct solutions across many tasks and domains. Further, it allows the luxury to scale without additional human effort, i.e., many additional problems can be created automatically.



**Title:** ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning
**Pdf:** https://arxiv.org/abs/2503.24378
**Task:** `acp_bench_hard`
**Abstract:**

We introduce ACPBench Hard, a dataset of generative, open-ended questions which LLM models needs to answer in order to plan. Models that perform well on these tasks could in principle be integrated into a planner or be used directly as a policy. We discuss the complexity of these tasks as well as the complexity of validating the correctness of their answers and present validation algorithms for each task. Equipped with these validators, we test the performance of a variety of models on our tasks and find that for most of these tasks, the performance of even the largest models is still subpar. Our experiments show that no model outperforms any other in these tasks, and with a few exceptions, all tested language models score below 65\%, indicating that even the current frontier language models as well as so-called reasoning models have a long way to go before they can reliably reason about planning.

The dataset is available on [HuggingFace](https://huggingface.co/datasets/ibm-research/acp_bench).


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

@misc{KokelKSS25ACPHard,
  title       = {ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning},
  author      = {Harsha Kokel and
                 Michael Katz and
                 Kavitha Srinivas and
                 Shirin Sohrabi},
  year        = {2025},
  eprint      = {2503.24378},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url         = {https://arxiv.org/abs/2503.24378},
}
```

### Groups, Tags, and Tasks

#### Groups

* None

#### Tags

* `acp_bench` : Evaluates `acp_bool_cot_2shot` and `acp_mcq_cot_2shot` (Main variant for ACPBench paper)
* `acp_bool_cot_2shot` : Evaluates `acp_areach_bool`, `acp_app_bool`, `acp_just_bool`, `acp_land_bool`, `acp_prog_bool`, `acp_reach_bool`, `acp_val_bool` with chain-of-thought and 2 shots
* `acp_mcq_cot_2shot` : Evaluates `acp_areach_mcq`, `acp_app_mcq`, `acp_just_mcq`, `acp_land_mcq`, `acp_prog_mcq`, `acp_reach_mcq`, `acp_val_mcq`  with chain-of-thought and 2 shots
* `acp_bench_hard` : Evaluates `acp_gen_2shot` (Main variant for ACPBench Hard paper)
* `acp_gen_2shot` : Evaluates `acp_areach_gen`, `acp_app_gen`, `acp_just_gen`, `acp_land_gen`, `acp_nexta_gen`, `acp_prog_gen`, `acp_reach_gen`, `acp_val_gen` with 2 shots
* `acp_bench_hard_with_pddl` : Evaluates `acp_gen_2shot_with_pddl`
* `acp_gen_2shot_with_pddl` : Evaluates `acp_areach_gen_with_pddl`, `acp_app_gen_with_pddl`, `acp_just_gen_with_pddl`, `acp_land_gen_with_pddl`, `acp_nexta_gen_with_pddl`, `acp_prog_gen_with_pddl`, `acp_reach_gen_with_pddl`, `acp_val_gen_with_pddl` with 2 shots

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

8 Generative tasks (with just natural language description in context)
* `acp_areach_gen`
* `acp_app_gen`
* `acp_just_gen`
* `acp_land_gen`
* `acp_nexta_gen`
* `acp_prog_gen`
* `acp_reach_gen`
* `acp_val_gen`

and the same 8 generative tasks with natural language as well as the PDDL description of the domain and problem in context.
* `acp_areach_gen_with_pddl`
* `acp_app_gen_with_pddl`
* `acp_just_gen_with_pddl`
* `acp_land_gen_with_pddl`
* `acp_nexta_gen_with_pddl`
* `acp_prog_gen_with_pddl`
* `acp_reach_gen_with_pddl`
* `acp_val_gen_with_pddl`

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
* 05/13/2025 Adding ACPBench Hard tasks (with and without PDDL)
