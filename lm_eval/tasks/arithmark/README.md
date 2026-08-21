# ArithMark-2.0

### Paper / Dataset

Title: `ArithMark-2.0: A Benchmark for Integer Arithmetic Reasoning in Language Models`

Homepage: https://huggingface.co/datasets/AxiomicLabs/ArithMark-2.0

### Citation

```bibtex
@dataset{arithmark2024,
  author = {Axiomic Labs},
  title = {ArithMark-2.0},
  year = {2024},
  url = {https://huggingface.co/datasets/AxiomicLabs/ArithMark-2.0},
  publisher = {Hugging Face}
}
```

### Description

ArithMark-2.0 is a synthetic benchmark of 2,500 integer arithmetic problems designed to evaluate language model arithmetic reasoning capabilities. Problems span four difficulty levels (easy, medium, hard) and cover operations including addition, subtraction, multiplication, division, mixed operations, and parenthesized expressions.

Each instance presents an arithmetic expression (e.g., `59 + 45 =`) followed by four candidate answers. Models are evaluated via multiple-choice log-likelihood scoring, selecting the answer with the highest probability.

### Groups, Tags, and Tasks

#### Groups

* `arithmark`: Evaluates all difficulty levels together (weighted aggregate)

#### Tags

* `arithmark_tasks`: Includes `arithmark_default`, `arithmark_easy`, `arithmark_medium`, `arithmark_hard`

#### Tasks

* `arithmark_default`: Full benchmark (2,500 problems)
* `arithmark_easy`: Single-operation problems (addition, subtraction, multiplication, division)
* `arithmark_medium`: Two-operation and parenthesized two-operation problems
* `arithmark_hard`: Three-operation and parenthesized three-operation problems

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

### Changelog

* Version 1.0: Initial addition of ArithMark-2.0 task with difficulty-based subtasks.
