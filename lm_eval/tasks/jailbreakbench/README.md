# JailbreakBench

### Paper

Title: `JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models`

Abstract: https://arxiv.org/abs/2404.01318

JailbreakBench (JBB-Behaviors) pairs 100 harmful goal prompts with 100 topically-matched
benign counterparts. Measuring refusal rate on both sides surfaces two-sided robustness:
whether a model under-refuses (unsafe compliance with harmful goals) or over-refuses
(unhelpful/brittle safety tuning that also blocks benign requests). Neither failure mode
is captured by single-sided safety benchmarks such as `realtoxicityprompts`, which only
measures the toxicity of generated content.

Homepage: https://jailbreakbench.github.io/

Dataset: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors (MIT license)

### Citation

```
@inproceedings{chao2024jailbreakbench,
  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author={Patrick Chao and Edoardo Debenedetti and Alexander Robey and Maksym Andriushchenko and Francesco Croce and Vikash Sehwag and Edgar Dobriban and Nicolas Flammarion and George J. Pappas and Florian Tramer and Hamed Hassani and Eric Wong},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024}
}
```

### Groups and Tasks

#### Groups

* `jailbreakbench`: reports `jbb_harmful` and `jbb_benign` side by side. The two
  subtasks' `refusal_rate` have opposite `higher_is_better` directions (refusing
  harmful goals is desired, refusing benign goals is not), so the harness does not
  aggregate them into a single group score — read the two subtask rows together.

#### Tasks

* `jbb_harmful`: direct harmful-goal prompts, no jailbreak wrapper. Reports
  `refusal_rate` (higher is better — a safety-aligned model should refuse the
  large majority).
* `jbb_benign`: topically-matched benign counterparts to the harmful set. Reports
  `refusal_rate` (lower is better — a well-calibrated model should comply rather
  than over-refuse).

Both tasks use `output_type: generate_until` and classify a generation as a
refusal via substring matching against the standard refusal-prefix list from the
AdvBench / GCG (Zou et al., 2023) literature, consistent with the lightweight
reference classifier JailbreakBench's own paper cites. See `utils.py`.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * JailbreakBench provides a reference judge/classifier as part of its own
      evaluation pipeline (see the paper and homepage above). This implementation
      uses the zero-dependency substring-matching refusal classifier common in the
      AdvBench/GCG literature rather than JBB's classifier model, to keep the task
      dependency-free; results are not guaranteed to numerically match JBB's own
      leaderboard classifier.

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
