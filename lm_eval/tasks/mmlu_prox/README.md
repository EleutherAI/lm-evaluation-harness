# MMLU-ProX

### Paper

Title: `MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation`

Abstract: `Traditional benchmarks like MMLU and MMLU-Pro focus primarily on single-language evaluation, limiting their ability to assess language models in multilingual and culturally diverse contexts. To address this gap, we introduce MMLU-ProX, a comprehensive multilingual benchmark that builds upon MMLU-Pro by covering multiple typologically diverse languages with approximately 11,829 questions per language.`

Homepage: https://mmluprox.github.io/

### Citation

```bibtex
@misc{mmluprox,
      title={MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation},
      author={Weihao Xuan and Rui Yang and Heli Qi and Qingcheng Zeng and Yunze Xiao and Yun Xing and Junjue Wang and Huitao Li and Xin Li and Kunyu Yu and Nan Liu and Qingyu Chen and Douglas Teodoro and Edison Marrese-Taylor and Shijian Lu and Yusuke Iwasawa and Yutaka Matsuo and Irene Li},
      year={2025},
      eprint={2503.10497},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.10497},
}
```

### Groups and Tasks

#### Groups

* `mmlu_pro_{lang}`: 'All 14 subjects of the mmlu_pro_prox dataset in {lang}, evaluated following the methodology in mmlu_pro's original implementation'

Available lang:
- ar
- bn
- de
- en
- es
- fr
- hi
- ja
- ko
- pt
- sw
- th
- zh

#### Tasks

The following tasks evaluate subjects in the mmlu_prox dataset
- `mmlu_prox_{lang}_biology`
- `mmlu_prox_{lang}_business`
- `mmlu_prox_{lang}_chemistry`
- `mmlu_prox_{lang}_computer_science`
- `mmlu_prox_{lang}_economics`
- `mmlu_prox_{lang}_engineering`
- `mmlu_prox_{lang}_health`
- `mmlu_prox_{lang}_history`
- `mmlu_prox_{lang}_law`
- `mmlu_prox_{lang}_math`
- `mmlu_prox_{lang}_other`
- `mmlu_prox_{lang}_philosophy`
- `mmlu_prox_{lang}_physics`
- `mmlu_prox_{lang}_psychology`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
