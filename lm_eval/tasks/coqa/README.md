# CoQA

### Paper

Title: `CoQA: A Conversational Question Answering Challenge`

Abstract: https://arxiv.org/pdf/1808.07042.pdf

CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

Homepage: https://stanfordnlp.github.io/coqa/

### Citation

```
BibTeX-formatted citation goes here
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `coqa`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Scoring and task version

Version 4.0 preserves one reference answer per annotator, including repeated
answers. CoQA computes EM and token F1 by leaving out each annotator in turn,
maximizing over the remaining references, and averaging the resulting scores.
Deduplicating references before this calculation changes annotator agreement:
for `red, red, blue, blue`, a prediction of `red` should receive 1.0, not 0.5.
This follows the authors' [official evaluator](https://github.com/stanfordnlp/coqa-baselines/blob/693d05b39dc036a8ad4ef0f00e4e6068aafa9d02/scripts/evaluate-v1.0.py).
Scores from versions 3.0 and 4.0 are therefore not directly comparable.

The existing `EleutherAI/coqa` train/validation loading and conversation format
are retained. The evaluated turn's answer and alternative annotations are kept
out of the prompt; prior conversational answers remain intentional context.
Synthetic tests cover answer conversion, annotator ordering, prompt separation,
registered YAML loading, generation requests, aggregation and an offline mock
run through the evaluator. These tests do not measure model quality or establish
absence of pretraining contamination.

Run the focused tests from the repository root:

```bash
python -m pip install -e . pytest 'transformers==4.57.6' 'datasets==4.4.1'
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest tests/test_coqa.py -q
```

These versions isolate scoring tests from the separate Transformers 5.x metric
import incompatibility tracked in [#4435](https://github.com/EleutherAI/lm-evaluation-harness/issues/4435).
The fixtures are original synthetic text; no CoQA passages are redistributed.
CoQA passages have [domain-specific licenses](https://stanfordnlp.github.io/coqa/#license).
