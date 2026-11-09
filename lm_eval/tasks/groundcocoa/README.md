# GroundCocoa

### Paper

Title: `GroundCocoa: A Benchmark for Evaluating Compositional & Conditional Reasoning in Language Models`

Abstract: https://arxiv.org/abs/2404.04237

The rapid progress of large language models (LLMs) has seen them excel and frequently surpass human performance on standard benchmarks. This has enabled many downstream applications, such as LLM agents, to rely on their reasoning to address complex task requirements. However, LLMs are known to unexpectedly falter in simple tasks and under seemingly straightforward circumstances - underscoring the need for better and more diverse evaluation setups to measure their true capabilities. To this end, we choose to study compositional and conditional reasoning, two aspects that are central to human cognition, and introduce GroundCocoa - a lexically diverse benchmark connecting these reasoning skills to the real-world problem of flight booking. Our task involves aligning detailed user preferences with available flight options presented in a multiple-choice format. Results indicate a significant disparity in performance among current state-of-the-art LLMs with even the best performing model, GPT-4 Turbo, not exceeding 67% accuracy despite advanced prompting techniques.

Homepage: `https://osu-nlp-group.github.io/GroundCocoa/`


### Citation

```
@misc{kohli2025groundcocoabenchmarkevaluatingcompositional,
      title={GroundCocoa: A Benchmark for Evaluating Compositional & Conditional Reasoning in Language Models},
      author={Harsh Kohli and Sachin Kumar and Huan Sun},
      year={2025},
      eprint={2404.04237},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2404.04237},
}
```

### Groups and Tasks

#### Groups

- Not part of a group yet

#### Tasks

- `groundcocoa`


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
