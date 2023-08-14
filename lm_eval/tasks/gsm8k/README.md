# GSM8k

## Paper
Training Verifiers to Solve Math Word Problems
https://arxiv.org/abs/2110.14168

State-of-the-art language models can match human performance on many tasks, but
they still struggle to robustly perform multi-step mathematical reasoning. To
diagnose the failures of current models and support research, we introduce GSM8K,
a dataset of 8.5K high quality linguistically diverse grade school math word problems.
We find that even the largest transformer models fail to achieve high test performance,
despite the conceptual simplicity of this problem distribution.

NOTE: See the official implementation of the task:
    https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
for how to make use of the dataset's calculator annotations in your language
model's sample/generation function.

Homepage: https://github.com/openai/grade-school-math


## Citation
```
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Groups and Tasks

#### Groups

- `math_word_problems`
- `chain_of_thought`
- `self_consistency`

#### Tasks

- `gsm8k_yaml`
- `gsm8k_cot`: GSM8K with Chain-of-Thought
- `gsm8k_cot_self_consistency`: GSM8K with Chain-of-Thought and Self-Consistency

### Checklist

- [x] Is in Eval-harness v1.0 ?
- [ ] Has been checked for regression from v1.0?
- [ ] Has been checked for equivalence with original paper methodology?
- [ ] "Main" checked variant clearly denoted?

### Variant Wishlist

- [ ] Variant with Calculator (see https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py for example implementation)
- [ ] Using Verifiers
- [ ] Majority voting "without CoT"
