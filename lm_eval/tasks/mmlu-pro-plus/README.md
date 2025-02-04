# mmlu_pro_plus

### Paper

Title: `MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs`

Abstract: `Existing benchmarks for large language models (LLMs) increasingly struggle to differentiate between
top-performing models, underscoring the need for more challenging evaluation frameworks.
We introduce MMLU-Pro+, an enhanced benchmark building upon MMLU-Pro to assess shortcut
learning and higher-order reasoning in LLMs. By incorporating questions with multiple
correct answers across diverse domains, MMLU-Pro+ tests LLMs' ability to engage in complex
reasoning and resist simplistic problem-solving strategies. Our results show that
MMLU-Pro+ maintains MMLU-Pro's difficulty while providing a more rigorous test of
model discrimination, particularly in multi-correct answer scenarios.
We introduce novel metrics like shortcut selection ratio and correct pair identification
ratio, offering deeper insights into model behavior and anchoring bias.
Evaluations of six state-of-the-art LLMs reveal significant performance gaps,
highlighting variations in reasoning abilities and bias susceptibility.`

Homepage: https://github.com/asgsaeid/mmlu-pro-plus

### Citation

```bibtex
@article{taghanaki2024mmlu,
  title={MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs},
  author={Taghanaki, Saeid Asgari and Khani, Aliasgahr and Khasahmadi, Amir},
  journal={arXiv preprint arXiv:2409.02257},
  year={2024}
}
```

### Groups and Tasks

#### Groups

* `mmlu_pro_plus`: 'All 14 subjects of the mmlu_pro_plus dataset, evaluated following the methodology in mmlu's original implementation'

#### Tasks

The following tasks evaluate subjects in the mmlu_pro dataset
- `mmlu_pro_plus_biology`
- `mmlu_pro_plus_business`
- `mmlu_pro_plus_chemistry`
- `mmlu_pro_plus_computer_science`
- `mmlu_pro_plus_economics`
- `mmlu_pro_plus_engineering`
- `mmlu_pro_plus_health`
- `mmlu_pro_plus_history`
- `mmlu_pro_plus_law`
- `mmlu_pro_plus_math`
- `mmlu_pro_plus_other`
- `mmlu_pro_plus_philosophy`
- `mmlu_pro_plus_physics`
- `mmlu_pro_plus_psychology`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
