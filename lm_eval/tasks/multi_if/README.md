# Multi-IF Evaluation

*Note: Current implementation only supports the single-turn multilingual instructions.*

### Paper

Title: Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following
Abstract: https://arxiv.org/abs/2410.15553

We introduce Multi-IF, a new benchmark designed to assess LLMs' proficiency in following multi-turn and multilingual instructions. Multi-IF, which utilizes a hybrid framework combining LLM and human annotators, expands upon the IFEval by incorporating multi-turn sequences and translating the English prompts into another 7 languages, resulting in a dataset of 4501 multilingual conversations, where each has three turns. Our evaluation of 14 state-of-the-art LLMs on Multi-IF reveals that it presents a significantly more challenging task than existing benchmarks. All the models tested showed a higher rate of failure in executing instructions correctly with each additional turn. For example, o1-preview drops from 0.877 at the first turn to 0.707 at the third turn in terms of average accuracy over all languages. Moreover, languages with non-Latin scripts (Hindi, Russian, and Chinese) generally exhibit higher error rates, suggesting potential limitations in the models’ multilingual capabilities.

Homepage: https://github.com/facebookresearch/Multi-IF/tree/main


### Citation

```
@article{he2024multi,
  title={Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following},
  author={He, Yun and Jin, Di and Wang, Chaoqi and Bi, Chloe and Mandyam, Karishma and Zhang, Hejia and Zhu, Chen and Li, Ning and Xu, Tengyu and Lv, Hongjiang and others},
  journal={arXiv preprint arXiv:2410.15553},
  year={2024}
}
```
