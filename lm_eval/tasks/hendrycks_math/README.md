# MATH

## Paper
Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/abs/2103.03874

Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations.

NOTE: This task corresponds to the MATH (`hendrycks_math`) implementation at https://github.com/EleutherAI/lm-evaluation-harness/tree/master . For the variant which uses the custom 4-shot prompt in the Minerva paper (https://arxiv.org/abs/2206.14858), and SymPy answer checking as done by Minerva, see `lm_eval/tasks/minerva_math`.

Homepage: https://github.com/hendrycks/math

## Changes

### v1.1
- Answer extraction now tries `\boxed{}` from the model response before falling back to `$...$` delimiters. This aligns with the `aime` task behavior and fixes zero accuracy for models that output `\boxed{}` formatted answers.
- Support for multiple `\boxed{}` answers (e.g. `\boxed{3}, \boxed{5}, \boxed{7}` is extracted as `3, 5, 7`). Duplicate boxes are deduplicated.
- Optional SymPy symbolic equivalence fallback and `math_verify` metric (see below).

## Optional enhanced equivalence (`pip install lm-eval[math]`)

When `sympy` is installed, `is_equiv` falls back to symbolic comparison via `sympy.simplify` when string matching fails. This handles cases like `9a + 11` vs `11 + 9a` or `\dfrac{1}{2}` vs `\frac{1}{2}` that are mathematically equivalent but differ as strings. Without `sympy`, the task uses string-only comparison (original behavior).

When `math_verify` is installed, an additional `math_verify` metric is reported alongside `exact_match`, providing robust LaTeX-aware answer verification (same as in `minerva_math`).


## Citation
```
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```

### Groups and Tasks

#### Groups

- `hendrycks_math`: the MATH benchmark from Hendrycks et al. 0- or few-shot.

#### Tasks

- `hendrycks_math_algebra`
- `hendrycks_math_counting_and_prob`
- `hendrycks_math_geometry`
- `hendrycks_math_intermediate_algebra`
- `hendrycks_math_num_theory`
- `hendrycks_math_prealgebra`
- `hendrycks_math_precalc`
- `hendrycks_math500`

### Checklist

The checklist is the following:

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * Answer extraction code is taken from the original MATH benchmark paper's repository.


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
