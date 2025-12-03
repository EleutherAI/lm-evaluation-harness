# SciKnowEval_mcqa

This task was submitted at the [NeurIPS 2025 E2LM](https://e2lmc.github.io/) competition, and reached $3^{rd}$ place on the general leaderboard.
Its intended use is within the context of Small Language Model (SLM) evaluation in early training stages. More details are provided in the competition [proposal paper](https://arxiv.org/pdf/2506.07731).

### Benchmark details

This task uses a subset of the [SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval) dataset. Specifically, it filters out non-MCQA samples and focuses on questions from levels L1, L2, and L3, which are designed to assess knowledge memory, comprehension and reasoning respectively, as described in the original [paper](https://arxiv.org/pdf/2406.09098v2).

The full SciKnowEval dataset is a comprehensive benchmark for evaluating the scientific knowledge reasoning capabilities of Large Language Models (LLMs). It spans across a few scientific domains: Physics, Chemistry, Biology and Materials.

SciKnowEval_mcqa dataset: https://huggingface.co/datasets/ShAIkespear/SciKnowEval_mcqa

### Citation

```
@misc{sci-know-2025-mcqa,
    title = "SciKnowEval_mcqa: A Benchmark for Small Language Model Evaluation in their Early Training Stages",
    author = "Anthony Kalaydjian, Eric Saikali",
    year = "2025",
}
```

### Groups and Tasks

#### Groups

* `sciknoweval_mcqa`: Evaluates `sciknoweval_Biology`, `sciknoweval_Chemistry`, `sciknoweval_Materials` and `sciknoweval_Physics`

#### Tasks
* `sciknoweval_Biology`: Data across all remaining splits corresponding to Biology MCQs.
* `sciknoweval_Chemistry`: Data across all remaining splits corresponding to Chemistry MCQs.
* `sciknoweval_Materials`: Data across all remaining splits corresponding to Materials MCQs.
* `sciknoweval_Physics`: Data across all remaining splits corresponding to Physics MCQs.

### Checklist

For adding novel benchmarks/datasets to the library:
  * [ ] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
