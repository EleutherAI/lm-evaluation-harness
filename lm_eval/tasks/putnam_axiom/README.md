# Putnam-AXIOM

## Paper

Title: `Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs`

Abstract: https://arxiv.org/abs/2508.08292 (ICML 2025, PMLR 267:20723-20747)

The **Putnam-AXIOM benchmark** is a new evaluation suite designed to measure mathematical reasoning capabilities of large language models (LLMs). It consists of:

* **Putnam-AXIOM Original**: 522 problems sourced from historical William Lowell Putnam Mathematical Competitions, curated to test advanced problem-solving abilities.
* **Putnam-AXIOM Variation**: functionally equivalent rewrites of 100 of the original problems, generated through programmatic transformations (e.g., modifying variables, constants, and surface features) to preserve difficulty while mitigating contamination from publicly available datasets.

The variation protocol can generate unboundedly many snapshots of those 100 problems. The
published evaluation (§3.3) uses **five snapshots**, and the Hugging Face `variations` split
materializes exactly those: 100 problems x 5 snapshots = 500 rows.

Homepage: https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522

### Citation

```bibtex
@InProceedings{pmlr-v267-gulati25a,
  title = {Putnam-{AXIOM}: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in {LLM}s},
  author = {Gulati, Aryan and Miranda, Brando and Chen, Eric and Xia, Emily and Fronsdal, Kai and De Moraes Dumont, Bruno and Koyejo, Sanmi},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages = {20723--20747},
  year = {2025},
  volume = {267},
  series = {Proceedings of Machine Learning Research},
  month = {13--19 Jul},
  publisher = {PMLR}
}
```

### Groups, Tags, and Tasks

#### Tasks

* `putnam_axiom_original`: The full set of 522 original Putnam problems (**main variant**). Matches the paper's Original evaluation, which is run once over all 522.
* `putnam_axiom_variations`: 500 docs — the 100 varied problems across all five published snapshots. Designed for robustness testing and contamination mitigation.
* `putnam_axiom_variations_org`: The 100 unmodified counterparts of the varied problems, i.e. the paired "Original" column the paper compares the variations against.
* `putnam_axiom`: Alias of `putnam_axiom_original`, for convenience.

All four use the paper's 4-shot prompt and its MATH-style SymPy equivalence check for
`exact_match`.

Because every snapshot covers the same 100 problems, the mean `exact_match` over the 500
`putnam_axiom_variations` docs equals the paper's reported mean over its five trials.

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
