# Task-name

### Paper

Title: `Measuring Massive Multitask Language Understanding`

Abstract: `https://arxiv.org/abs/2009.03300`

`The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.`

Homepage: `https://github.com/hendrycks/test`

Note: The `Flan` variants are derived from [here](https://github.com/jasonwei20/flan-2), and as described in Appendix D.1 of [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416).

### Citation

```
@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

### Groups, Tags, and Tasks

#### Groups

* `mmlu`: `Original multiple-choice MMLU benchmark`
* `mmlu_continuation`: `MMLU but with continuation prompts`
* `mmlu_generation`: `MMLU generation`

MMLU is the original benchmark as implemented by Hendrycks et al. with the choices in context and the answer letters (e.g `A`, `B`, `C`, `D`) in the continuation.
`mmlu_continuation` is a cloze-style variant without the choices in context and the full answer choice in the continuation.
`mmlu_generation` is a generation variant, similar to the original but the LLM is asked to generate the correct answer letter.


#### Subgroups

* `mmlu_stem'
* `mmlu_humanities'
* `mmlu_social_sciences'
* `mmlu_other'

Subgroup variants are prefixed with the subgroup name, e.g. `mmlu_stem_continuation`.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

# changelog
ver 1: PR #497
switch to original implementation

ver 2: PR #2116
add missing newline in description.
