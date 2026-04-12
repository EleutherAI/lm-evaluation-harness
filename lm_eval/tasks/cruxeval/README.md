# CRUXEval

## Paper

CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution

https://arxiv.org/abs/2401.03065

CRUXEval is a benchmark of 800 Python functions paired with input-output pairs. It evaluates a model's ability to reason about code execution in two directions: given a function and input, predict the output (`cruxeval_output`); and given a function and output, find a valid input (`cruxeval_input`). The benchmark includes chain-of-thought variants that prompt the model to reason step by step before answering.

Homepage: https://crux-eval.github.io

## Citation

```
@article{gu2024cruxeval,
  title={CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution},
  author={Alex Gu and Baptiste Rozière and Hugh Leather and Armando Solar-Lezama and Gabriel Synnaeve and Sida I. Wang},
  year={2024},
  eprint={2401.03065},
  archivePrefix={arXiv},
  primaryClass={cs.SE}
}
```

#### Tasks

- `cruxeval_input`: Input prediction — given a function and its output, predict a valid input. pass@1 at temperature 0.2. Matches the published evaluation setup from the paper.
- `cruxeval_input_08`: Input prediction at temperature 0.8 for pass@5 evaluation. Matches the published pass@5 setup from the paper.
- `cruxeval_input_cot`: Input prediction with chain-of-thought reasoning prompting.
- `cruxeval_output`: Output prediction — given a function and its input, predict the output. pass@1 at temperature 0.2. Matches the published evaluation setup from the paper.
- `cruxeval_output_08`: Output prediction at temperature 0.8 for pass@5 evaluation. Matches the published pass@5 setup from the paper.
- `cruxeval_output_cot`: Output prediction with chain-of-thought reasoning prompting.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

v1.0: Initial implementation with input/output prediction tasks and chain-of-thought variants.
