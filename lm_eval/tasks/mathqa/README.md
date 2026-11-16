# MathQA

### Paper

MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms
https://arxiv.org/pdf/1905.13319.pdf

MathQA is a large-scale dataset of 37k English multiple-choice math word problems
covering multiple math domain categories by modeling operation programs corresponding
to word problems in the AQuA dataset (Ling et al., 2017).

Homepage: https://math-qa.github.io/math-QA/


### Citation

```
@misc{amini2019mathqa,
    title={MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms},
    author={Aida Amini and Saadia Gabriel and Peter Lin and Rik Koncel-Kedziorski and Yejin Choi and Hannaneh Hajishirzi},
    year={2019},
    eprint={1905.13319},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `math_word_problems`

#### Tasks

* `mathqa`: The MathQA dataset, as a multiple choice dataset where the answer choices are not in context.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The MathQA dataset predates transformer-based prompted LLMs. We should, however, return to this task to ensure equivalence to the non-CoT version of mathQA used in the Chain-of-Thought paper.

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  * [x] Checked for equivalence with v0.3.0 LM Evaluation Harness
