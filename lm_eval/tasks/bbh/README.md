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

### Groups and Tasks

#### Groups

- `bbh_zeroshot`
- `bbh_fewshot`
- `bbh_cot_fewshot`
- `bbh_cot_zeroshot`


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
