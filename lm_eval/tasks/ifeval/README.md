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

- version 4.1 (15-Nov-2026); scoring is deterministic regardless of item order. When the dataset leaves a
  checker's arguments null, `build_description()` fills them from `random`; those draws came from the
  process-wide RNG, so the same (prompt, response) pair could score both True and False depending on how many
  items were scored before it. They are now drawn from a generator seeded per instruction from the item's own
  content, and scoring no longer consumes the global stream. Verdicts for items with null arguments can differ
  from historical unseeded runs (#4602). On the in-tree `google/IFEval` data only two instructions draw anything:
  key 1122 (`#` at least 4, now drawn as `h` at least 4) and key 1129 (`!` at least 6, now drawn as `e` at least
  6), so those two are the only items whose verdicts can move between 4.0 and 4.1.
- version 1.1 of `ifeval_es` and `ifeval_ca`; the same per-item seeding now covers the Spanish and Catalan
  scorers. They draw the same two keys (es: `j` at least 4 and `r` at least 6; ca: `k` at least 4 and `f` at
  least 6). Those checkers also draw the `relation` argument for number_highlighted_sections,
  number_placeholders and multiple_sections, but check_following never reads it, so no verdict depends on it.
