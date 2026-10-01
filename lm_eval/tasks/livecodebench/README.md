# LiveCodeBench

### Paper

LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code
https://arxiv.org/abs/2403.07974

LiveCodeBench is a code-generation benchmark scraped continuously from LeetCode, AtCoder, and Codeforces, and partitioned into release windows by contest date to enable contamination-free evaluation. This task uses the `code_generation_lite` subset, which removes problems whose test cases are unsuitable for the execution-based checker.

Scoring is execution-based. Generated code is extracted from the first fenced code block in the model output and run against the problem's public and private test cases using the benchmark's official checker (`testing_util.py`, vendored under the MIT license). Standard-input problems are scored by running the program with mocked stdin/stdout. Call-based problems are scored by invoking the `Solution` class method named in the problem metadata. A problem passes only if every test case passes (`pass@1`, mean over problems).

This task executes model-generated code. It is marked `unsafe_code` and requires `--confirm-run-unsafe-code` to run. The checker's `reliability_guard` is a best-effort guard, not a sandbox.

### Citation

```text
@article{jain2024livecodebench,
  title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  author={Jain, Naman and Han, King and Gu, Alex and Li, Wen-Ding and Yan, Fanjia and Zhang, Tianjun and Wang, Sida and Solar-Lezama, Armando and Sen, Koushik and Stoica, Ion},
  journal={arXiv preprint arXiv:2403.07974},
  year={2024}
}
```

### Groups, Tags, and Tasks

#### Tasks

All windows are cumulative and start on 2023-05-07; each release extends the end date.

* `livecodebench`: `release_v6, 1055 problems, through 2025-04-06`
* `livecodebench_v1`: `release_v1, 400 problems, through 2024-03-02`
* `livecodebench_v2`: `release_v2, 511 problems, through 2024-05-25`
* `livecodebench_v3`: `release_v3, 612 problems, through 2024-08-10`
* `livecodebench_v4`: `release_v4, 713 problems, through 2024-10-05`
* `livecodebench_v5`: `release_v5, 880 problems, through 2025-01-04`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The checker is vendored from the LiveCodeBench repository (`lcb_runner/evaluation/testing_util.py`), with two documented deviations. Private test cases are decoded through a restricted unpickler that accepts only plain data, and `reliability_guard` is applied once per process because it is not idempotent.

* [x] Is the task in the correct directory? If it is a subfolder in a larger benchmark, it should be nested appropriately.
* [ ] Have you confirmed the test split being used?
  * All release windows load from the `test` split.

### Variant Wishlist

* `livecodebench_v6` and later windows as upstream releases land.
* `livecodebench_instruct` variant matching the instruction-tuned prompt format used by the official LiveCodeBench runner.
* Multi-sample `pass@k` scoring (`repeats: k`, k > 1) matching the official estimator.
