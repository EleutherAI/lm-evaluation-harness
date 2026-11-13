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

The current task versions preserve all non-`None` instruction arguments in both
strict and loose evaluation, including zero-valued counts. For example, a
requirement for exactly zero bullet points remains zero rather than being replaced
with a randomly chosen positive count. This matters for inputs that contain such
arguments, including custom datasets.

#### Version history

* [2026-11-13] [PR #4591](https://github.com/EleutherAI/lm-evaluation-harness/pull/4591):
  `ifeval` 4.0 -> 5.0, `leaderboard_ifeval` 3.0 -> 4.0, and the Spanish and Catalan
  variants 1.0 -> 2.0 preserve non-`None` arguments. The version increments follow
  the [task versioning guidance](../../../docs/new_task_guide.md#versioning) and
  identify the evaluator change; they do not indicate improved model capability.

#### Dataset compatibility

[HarshShroff's review of PR #4591](https://github.com/EleutherAI/lm-evaluation-harness/pull/4591#pullrequestreview-5892631756)
reported no zero-valued or other falsy non-`None` instruction arguments in the
`google/IFEval`, `wis-k/instruction-following-eval`, `BSC-LT/IFEval_es`, and
`projecte-aina/IFEval_ca` data used by these tasks. On those data snapshots, the old
and new filters therefore pass the same arguments to the checkers, so this
correction does not change scores with the same responses and random state.
The zero-count regressions exercise additional valid inputs; they are not a claim
of a score change on the reviewed dataset snapshots.

#### Known upstream behavior

For a recorded `letter` of `'#'` or `'!'`, `LetterFrequencyChecker` selects a
random ASCII letter and lowercases it. The checker consequently counts that
letter rather than the recorded punctuation, and the result can depend on the
random state. This matches
[Google's reference checker](https://github.com/google-research/google-research/blob/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval/instructions.py#L1316-L1392)
and is also present in the Spanish and Catalan checkers. These nonempty character
arguments pass both the old and new argument filters, so the zero-count correction
does not alter this behavior. Counting recorded nonletter characters would be a
separate evaluator change requiring its own versioning and compatibility review.

#### Groups

* Not part of a group yet

#### Tasks

* `ifeval`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
