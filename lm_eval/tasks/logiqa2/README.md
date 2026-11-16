# LogiQA 2.0

### Paper

LogiQA 2.0 — An Improved Dataset for Logical Reasoning in Natural Language Understanding https://ieeexplore.ieee.org/document/10174688


The dataset is an amendment and re-annotation of LogiQA in 2020, a large-scale logical reasoning reading comprehension dataset adapted from the Chinese Civil Service Examination. This new version has an increased data size, the texts are refined with manual translation by professionals, and improved by removing items with distinctive cultural features like Chinese idioms.

Furthermore, a two-way natural language inference (NLI) task is introduced, resulting in 35k premise-hypothesis pairs with gold labels, making it the first large-scale NLI dataset for complex logical reasoning

Homepage: https://github.com/csitfun/LogiQA2.0

### Citation

```bibtex
@ARTICLE{10174688,
  author={Liu, Hanmeng and Liu, Jian and Cui, Leyang and Teng, Zhiyang and Duan, Nan and Zhou, Ming and Zhang, Yue},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title={LogiQA 2.0 — An Improved Dataset for Logical Reasoning in Natural Language Understanding},
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TASLP.2023.3293046}}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `logiqa2_zh`: The original dataset in Chinese.
* `logiqa2_NLI`: The NLI version of the dataset converted from the MRC version.
* `logieval`: Prompt based; https://github.com/csitfun/LogiEval

NOTE! The subtasks have not been verified yet.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] The original paper does not. There is another implementation of this task, but it designed for instruction tuned models: https://github.com/csitfun/LogiEval

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
