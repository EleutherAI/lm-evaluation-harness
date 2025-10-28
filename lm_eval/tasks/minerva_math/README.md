# MATH

ℹ️ This is the 4-shot variant!

## Paper

Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/abs/2103.03874

Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of
computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging
competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach
models to generate answer derivations and explanations.

NOTE: The few-shot and the generated answer extraction is based on the [Minerva](https://arxiv.org/abs/2206.14858) and
exact match equivalence is calculated using the `sympy` library. This requires additional dependencies, which can be
installed via the `lm-eval[math]` extra.

Homepage: https://github.com/hendrycks/math

## Citation

```
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}

@misc{2206.14858,
Author = {Aitor Lewkowycz and Anders Andreassen and David Dohan and Ethan Dyer and Henryk Michalewski and Vinay Ramasesh and Ambrose Slone and Cem Anil and Imanol Schlag and Theo Gutman-Solo and Yuhuai Wu and Behnam Neyshabur and Guy Gur-Ari and Vedant Misra},
Title = {Solving Quantitative Reasoning Problems with Language Models},
Year = {2022},
Eprint = {arXiv:2206.14858},
}
```

### Groups and Tasks

#### Groups

- `minerva_math`

#### Tasks

- `minerva_math_algebra`
- `minerva_math_counting_and_prob`
- `minerva_math_geometry`
- `minerva_math_intermediate_algebra`
- `minerva_math_num_theory`
- `minerva_math_prealgebra`
- `minerva_math_precalc`

### Checklist

The checklist is the following:

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
    * [x] Have you referenced the original paper that introduced the task?
    * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the
      reference implementation and documented how to run such a test?
        * The implementation in the original paper is one where the model is first fine-tuned on the data. They do have
          a few-shot evaluation for GPT-3, however the few-shot context used here is sourced
          from [Lewkowycz et al](https://arxiv.org/abs/2206.14858). The achieved accuracy on Llama-2 models is
          comparable to that provided in the paper, though not identical.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Variant Wishlist

- [ ] zero-shot variant

### Changelog

- version 2.0: (21-Feb-2025); added math_verify (extraction) metric. For
  details [see](https://huggingface.co/blog/math_verify_leaderboard)
- version 3.0 (21-Aug-2025); pass the full solution and model generation to `math_verify`'s `parse`
