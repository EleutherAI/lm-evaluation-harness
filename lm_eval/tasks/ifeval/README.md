# IFEval

### Paper

Title: Instruction-Following Evaluation for Large Language Models
Abstract: https://arxiv.org/abs/2311.07911

One core capability of Large Language Models (LLMs) is to follow natural language instructions. However, the evaluation of such abilities is not standardized: Human evaluations are expensive, slow, and not objectively reproducible, while LLM-based auto-evaluation is potentially biased or limited by the ability of the evaluator LLM. To overcome these issues, we introduce Instruction-Following Eval (IFEval) for large language models. IFEval is a straightforward and easy-to-reproduce evaluation benchmark. It focuses on a set of "verifiable instructions" such as "write in more than 400 words" and "mention the keyword of AI at least 3 times". We identified 25 types of those verifiable instructions and constructed around 500 prompts, with each prompt containing one or more verifiable instructions. We show evaluation results of two widely available LLMs on the market. Our code and data can be found at https://github.com/google-research/google-research/tree/master/instruction_following_eval

Homepage: https://github.com/google-research/google-research/tree/master/instruction_following_eval


### Citation

```
@article{zhou2023instructionfollowing,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and Siddhartha Brahma and Sujoy Basu and Yi Luan and Denny Zhou and Le Hou},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023},
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `ifeval`

### Scoring changes

* 2026-09-23: `ifeval` 5.0 and `leaderboard_ifeval` 4.0 use a task-local
  langdetect factory with seed 0 for the response-language and English case
  checkers. Ambiguous responses may score differently from earlier unseeded
  runs. This follows the language-seeding proposal in
  [Google PR #3428](https://github.com/google-research/google-research/pull/3428)
  (not yet merged), without its separate letter-frequency changes. See
  [Harness issue #4213](https://github.com/EleutherAI/lm-evaluation-harness/issues/4213)
  for the reproduction and the [junekihong.com benchmark author's report](https://junekihong.com/benchmarks/).

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
