# HMMT

### Paper

The Harvard-MIT Mathematics Tournament (HMMT) is a high-school mathematics
competition. Each event contains 30 short-answer problems whose gold answers
are integers or short closed-form expressions, making it a close sibling of
the AIME benchmark already supported by the harness.

These tasks use the problem sets curated and answer-verified by
[MathArena](https://matharena.ai/) ([eth-sri/matharena](https://github.com/eth-sri/matharena)),
mirroring the AIME task setup but scoring answers with `math-verify` in
addition to the AIME-style string-normalization exact-match (see the request in
[issue #3557](https://github.com/EleutherAI/lm-evaluation-harness/issues/3557)).

Homepage: https://matharena.ai/

### Citation

```text
@article{dekoninck2026matharena,
      title={Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs},
      author={Jasper Dekoninck and Nikola Jovanović and Tim Gehrunger and Kári Rögnvaldsson and Ivo Petrov and Chenhao Sun and Martin Vechev},
      year={2026},
      eprint={2605.00674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.00674}
}
```

The underlying datasets (`MathArena/hmmt_feb_2023`, `MathArena/hmmt_feb_2024`,
`MathArena/hmmt_feb_2025`, `MathArena/hmmt_nov_2025`) are distributed under the
CC BY-NC-SA 4.0 license. Please abide by the license when using the data.

### Groups, Tags, and Tasks

#### Tags

* `hmmt`: runs all four per-period HMMT tasks together (mirrors how `aime` groups its
  per-year tasks with a tag rather than an aggregating group). Each task reports its
  own `exact_match` / `math_verify`; no cross-task aggregate metric is defined.
* `math_word_problems`

#### Tasks

* `hmmt_feb_2023`: `HMMT February 2023 problems`
* `hmmt_feb_2024`: `HMMT February 2024 problems`
* `hmmt_feb_2025`: `HMMT February 2025 problems`
* `hmmt_nov_2025`: `HMMT November 2025 problems`

### Metrics

Each task reports two metrics:

* `exact_match`: AIME-style answer extraction (prefers a `\boxed{...}` answer,
  falling back to the last `$...$` span) followed by LaTeX string-normalization
  equivalence (copied from `lm_eval/tasks/aime/utils.py`).
* `math_verify`: symbolic equivalence between the gold answer and the model
  response via the `math-verify` library, which catches answers that are
  mathematically equal but not string-identical (e.g. fractions, surds).

`math-verify` is only available in the `[math]` extra:

```sh
pip install -e ".[math]"
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
