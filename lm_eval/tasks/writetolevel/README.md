# WriteToLevel

### Paper

Title: Which HSK 3.0? Two Official Documents, and Half of All Grading Decisions Change (Section: WriteToLevel)

Can a model write Chinese at the proficiency level it was asked for? WriteToLevel has 150 tasks: six target levels (HSK 1–6) × 25 topics, five of them held out. Each asks for a passage in Simplified Chinese on a topic, readable at a given level of the HSK 3.0 examination syllabus, at roughly a given length.

The grader is the metric, not the label. A passage hits its target when at least 95% of its characters come from the target level or below, measured by [hsk30](https://github.com/harukicoder/hsk30) against the 2025 HSK examination syllabus. No judge model is involved, so results are exactly reproducible.

Homepage: https://github.com/harukicoder/hsk30/tree/main/benchmark
Dataset: https://huggingface.co/datasets/harukicoder/writetolevel

### Citation

```
@software{serrano2026hsk30,
  author = {Serrano, Alvaro},
  title  = {hsk30: grading Chinese text against either document called HSK 3.0, with the WriteToLevel benchmark},
  year   = {2026},
  doi    = {10.5281/zenodo.22234657},
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `writetolevel`: all 150 tasks. Metrics:
  * `level_accuracy` (headline): measured level ≤ target.
  * `signed_error`: measured − target, in levels. Text that reaches no level counts as one step beyond the 7–9 band.
  * `length_compliance`: within the task's character tolerance.
  * `budget_violation`: one character alone exceeds the 5% above-target budget.

Requires `pip install lm_eval[writetolevel]`, which installs `hsk30`, a dependency-free package.

Read the per-level results, not only the aggregate: HSK 1–2 is where systems differ. At HSK 5–6 almost any fluent Chinese meets the 95% bar.

### Reference results

Temperature 0, scored by the reference implementation:

| System | level_accuracy | signed_error |
|---|---|---|
| deepseek-chat | 0.6667 | 0.24 |
| qwen3-235b | 0.5600 | 0.41 |
| gpt-4o | 0.5267 | 0.51 |

`utils.process_results` reproduces these exactly from the published outputs; `benchmark/lm_eval/writetolevel/check_writetolevel.py` in the hsk30 repository runs that check.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
