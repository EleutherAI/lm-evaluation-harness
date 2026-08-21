# COLLIE

### Paper

Title: COLLIE: Systematic Construction of Constrained Text Generation Tasks
Abstract: https://arxiv.org/abs/2307.08689

Text generation under constraints have seen increasing interests in natural language processing, especially with the rapidly improving capabilities of large language models. However, existing benchmarks for constrained generation usually focus on fixed constraint types (e.g.,generate a sentence containing certain words) that have proved to be easy for state-of-the-art models like GPT-4. We present COLLIE, a grammar-based framework that allows the specification of rich, compositional constraints with diverse generation levels (word, sentence, paragraph, passage) and modeling challenges (e.g.,language understanding, logical reasoning, counting, semantic planning). We also develop tools for automatic extraction of task instances given a constraint structure and a raw text corpus. Using COLLIE, we compile the COLLIE-v1 dataset with 2080 instances comprising 13 constraint structures. We perform systematic experiments across five state-of-the-art instruction-tuned language models and analyze their performances to reveal shortcomings. COLLIE is designed to be extensible and lightweight, and we hope the community finds it useful to develop more complex constraints and evaluations in the future.

Homepage: https://github.com/princeton-nlp/Collie


### Citation

```
@misc{yao2023collie,
      title={{COLLIE}: Systematic Construction of Constrained Text Generation Tasks},
      author={Shunyu Yao and Howard Chen and Austin W. Hanjie and Runzhe Yang and Karthik Narasimhan},
      year={2023},
      eprint={2307.08689},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `collie`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * Scoring reuses the upstream constraint checker directly. `constraints.py` is from `github.com/princeton-nlp/collie` and the official `all_data.dill` is loaded as-is.


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
