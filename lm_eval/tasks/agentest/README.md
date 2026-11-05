# agentest

Community task for the open [Agentic AI Test Checklist](https://huggingface.co/datasets/srinivaspadmanabhuni/agentic-ai-test-checklist-110) examples.

This is an open checklist of 110 tests across 4 pillars (evals, security, privacy, and responsible behaviour), with examples, labelled traces, and an open-source [agentest](https://github.com/srpadmanabhuni/agentest) runner. The catalogue is CC BY 4.0. This harness task is a community YAML config and does not depend on the agentest package.

`agentest_checklist_examples` loads Hugging Face dataset `srinivaspadmanabhuni/agentic-ai-test-checklist-110`, config `examples` (404 synthetic example rows: scenario, input, and expected behaviour). The full checklist is config `all` (110 tests). The only split is `train`, which this task uses as the evaluation split. The model is asked to state the expected behaviour for each scenario. The score is corpus BLEU against `expected_behavior`. That is a text-overlap score on the written examples. It does not execute agent traces; the agentest runner does that separately.

### Links

- Dataset: https://huggingface.co/datasets/srinivaspadmanabhuni/agentic-ai-test-checklist-110
- Runner: https://github.com/srpadmanabhuni/agentest
- DOI: https://doi.org/10.5281/zenodo.23597877
- Contact: srinivas AT testAIng.com

### Groups and Tasks

#### Groups

* `agentest`

#### Tasks

* `agentest_checklist_examples`: generative task over the `examples` config.

### Run

The task is registered under `lm_eval/tasks`, so `--include_path` is not required:

```bash
lm_eval --model dummy --tasks agentest_checklist_examples --limit 3
```

To run the group:

```bash
lm_eval --model dummy --tasks agentest --limit 3
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

The `examples` task is the variant added here. It evaluates whether a model can state the written expected behaviour for each checklist scenario. No published model-score setup is matched, because the catalogue is not a leaderboard.
