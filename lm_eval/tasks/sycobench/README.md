# SycoBench-600

### Paper

Title: `SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants`

ACL Anthology: https://aclanthology.org/2026.findings-acl.1759/

DOI: https://doi.org/10.18653/v1/2026.findings-acl.1759

Dataset: https://huggingface.co/datasets/dsinha/sycobench-600

Reference implementation: https://github.com/debu-sinha/sycobench-600

SycoBench-600 evaluates whether language model assistants preserve correct answers under user pressure and correct themselves when their initial answer is wrong. This lm-evaluation-harness task implements the single-turn multiple-choice baseline over the public Hugging Face dataset. The full multi-turn sycophancy and correction-selectivity protocol is implemented in the SycoBench Inspect task and is listed in the Inspect Evals Register.

Inspect Evals Register: https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/register/sycobench-600

### Citation

```
@inproceedings{sinha2026sycobench,
  title = {{SycoBench-600}: Measuring Sycophancy and Correction Selectivity in {LLM} Assistants},
  author = {Sinha, Debu},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year = {2026},
  pages = {35278--35284},
  doi = {10.18653/v1/2026.findings-acl.1759},
  url = {https://aclanthology.org/2026.findings-acl.1759/}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `sycobench_600_baseline`: zero-shot multiple-choice accuracy over the 600-item SycoBench-600 baseline dataset.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The reference implementation is available in the SycoBench repository. This lm-evaluation-harness task matches the baseline multiple-choice setting; the repository's Inspect task implements the full multi-turn protocol.


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
  * The lm-evaluation-harness variant is explicitly named `sycobench_600_baseline` because it is the single-turn baseline, not the full multi-turn protocol.
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  * This variant matches the baseline multiple-choice accuracy setup over the published 600-item dataset.
