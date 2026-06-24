# Putnam-AXIOM

## Paper

Title: `Putnam-AXIOM: A Benchmark for Evaluating Mathematical Reasoning in LLMs`

Abstract: https://openreview.net/forum?id=putnam-axiom (NeurIPS Workshop) and ICML 2025 submission

The **Putnam-AXIOM benchmark** is a new evaluation suite designed to measure mathematical reasoning capabilities of large language models (LLMs). It consists of:

* **Putnam-AXIOM Original**: 522 problems sourced from historical William Lowell Putnam Mathematical Competitions, curated to test advanced problem-solving abilities.
* **Putnam-AXIOM Variation**: 100 functionally equivalent problems generated through programmatic transformations (e.g., modifying variables, constants, and surface features) to preserve difficulty while mitigating contamination from publicly available datasets.

Homepage: https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522

### Citation

```text
@article{putnam-axiom-2025,
  title={Putnam-AXIOM: A Benchmark for Evaluating Mathematical Reasoning in LLMs},
  author={Xia, Emily and Miranda, Brando and others},
  year={2025}
}
```

### Groups, Tags, and Tasks

#### Tasks

* `putnam_axiom_original`: The full set of 522 original Putnam problems from historical competitions.
* `putnam_axiom_variations`: The 100 transformed variation problems, designed for robustness testing and contamination mitigation.
* `putnam_axiom_variations_org`: The original counterparts for each of the 100 variation problems, for comparison purposes.
* `putnam_axiom`: Base task configuration (defaults to `putnam_axiom_original`).

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* Initial implementation for Putnam-AXIOM benchmark suite
* Added variation tasks for robustness testing and contamination evaluation



