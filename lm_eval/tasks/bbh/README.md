# BigBenchHard

## Paper
Title: `Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them`
Abstract: https://arxiv.org/abs/2210.09261

A suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH).
These are the task for which prior language model evaluations did not outperform
the average human-rater.

Homepage: https://github.com/suzgunmirac/BIG-Bench-Hard


## Citation
```
@article{suzgun2022challenging,
  title={Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them},
  author={Suzgun, Mirac and Scales, Nathan and Sch{\"a}rli, Nathanael and Gehrmann, Sebastian and Tay, Yi and Chung, Hyung Won and Chowdhery, Aakanksha and Le, Quoc V and Chi, Ed H and Zhou, Denny and and Wei, Jason},
  journal={arXiv preprint arXiv:2210.09261},
  year={2022}
}
```

### Groups, Tags, and Tasks

#### Groups

- `bbh`: is the same as `bbh_cot_fewshot`.
- `bbh_zeroshot`
- `bbh_fewshot`
- `bbh_cot_fewshot`
- `bbh_cot_zeroshot`

#### Tags

None.

#### Tasks

- ...

### Checklist

- [x] Is in Eval-harness v1.0 ?
- [ ] Has been checked for regression from v1.0?
- [ ] Has been checked for equivalence with original paper methodology?
- [ ] "Main" checked variant clearly denoted?

### Variant Wishlist

- [ ] Variant with Calculator (see https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py for example implementation)
- [ ] Using Verifiers
- [ ] Majority voting "without CoT"

### Changelog
- no version change: changed dataset to `SaylorTwift/bbh`. Do not expect any change in the results.
- `bbh_cot_fewshot` v.4.0; 2025-07-14:
  - PR #3140. Removed duplicate "Let's think step by step" from the fewshots.
  - set target_delimiter to "" as the fewshot samples end with a newline character.
- `bbh_fewshot` tasks v3.0 and `bbh_cot_fewshot` tasks v5.0; groups `bbh_fewshot`, `bbh_cot_fewshot` and `bbh` v4.0; 2026-09-23:
  - The stop sequence `Q` is now `Q:`, as in `bbh_zeroshot` and `bbh_cot_zeroshot`. The bare `Q` cut each generation at its first capital Q, so the 7 `reasoning_about_colored_objects` questions whose answer is option `(Q)` could never be scored correct, and chains of thought that mention a word starting with Q before the answer were cut short.
