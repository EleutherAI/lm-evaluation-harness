# DziriEval

### Dataset

DziriEval is a multiple-choice benchmark for evaluating large language models on Algerian Arabic (Darja). It contains 1,000 four-option questions (A–D) authored directly in Darja rather than translated from English or MSA, covering 20 domains grouped into 5 categories: popular culture, geography and identity, history and memory, language and lexicon, and daily life. Questions include real Darja phenomena such as Arabizi and Darja–French code-switching.

The dataset ships as a single split of 1,000 rows, which is used in full as the evaluation set. The task is therefore zero-shot only.

Homepage: [https://huggingface.co/datasets/algerian-nlp/DziriEval](https://huggingface.co/datasets/algerian-nlp/DziriEval)

License: MIT

Note: the correct answers are not evenly distributed over the four options (A: 229, B: 290, C: 290, D: 191), so the majority-class baseline is 29.0% rather than 25%.

### Citation

```
@misc{algerian_nlp_dzirieval,
  title  = {DziriEval: 1,000 Algerian Darja multiple-choice questions},
  author = {Touati, Kamel and Algerian NLP Collective},
  year   = {2026},
  url    = {https://huggingface.co/datasets/algerian-nlp/DziriEval}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet.

#### Tasks

* `dzirieval`: all 1,000 questions, zero-shot. The prompt shows the question followed by the four lettered options, and the model is scored on the log-likelihood of the answer letter (MMLU-style).

### Results

Zero-shot accuracy on all 1,000 questions (± standard error):

| Model | acc |
|-------|-----|
| Majority-class baseline | 0.290 |
| Qwen/Qwen2.5-0.5B | 0.418 ± 0.016 |
| Qwen/Qwen2.5-1.5B | 0.516 ± 0.016 |

Reproduce with:

```
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B --tasks dzirieval --batch_size 8
```

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The benchmark has no accompanying paper yet; it is documented on its Hugging Face dataset card, which is cited above. No reference implementation or published scores exist to compare against.

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* Version 0.0: initial release.
