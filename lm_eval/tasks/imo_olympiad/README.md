# IMO Olympiad

## Paper

Title: `Putnam-AXIOM: A Benchmark for Evaluating Mathematical Reasoning in LLMs`

Abstract: https://openreview.net/forum?id=putnam-axiom (NeurIPS Workshop) and ICML 2025 submission

The IMO Olympiad benchmark evaluates mathematical problem-solving capabilities of large language models using International Mathematical Olympiad (IMO) style problems.

Homepage: https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-5272

### Citation

```text
@article{putnam-axiom-2025,
  title={Putnam-AXIOM: A Benchmark for Evaluating Mathematical Reasoning in LLMs},
  author={Xia, Emily and Miranda, Brando and others},
  year={2025}
}
```

### Groups, Tags, and Tasks

#### Groups

* `olympiad`: Mathematical olympiad-style problems

#### Tasks

* `olympiad_MM_maths`: Mathematics problems in multiple-choice format
* `olympiad_TO_maths`: Mathematics problems requiring direct answer computation
* `olympiad_MM_physics`: Physics problems in multiple-choice format
* `olympiad_TO_physics`: Physics problems requiring direct answer computation

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

