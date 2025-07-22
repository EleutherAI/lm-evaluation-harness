# MMLU-ProX

### Paper

Title: `MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation`

Abstract: `Existing large language model (LLM) evaluation benchmarks primarily focus on English, while current multilingual tasks lack parallel questions that specifically assess cross-linguistic reasoning abilities.
This dual limitation makes it challenging to comprehensively assess LLMs' performance in the multilingual setting. To fill this gap, we introduce MMLU-ProX, a comprehensive benchmark covering 29 languages, built on an English benchmark.
Each language version consists of 11,829 identical questions, enabling direct cross-linguistic comparisons. Additionally, to meet efficient evaluation needs, we provide a lite version containing 658 questions per language.
To ensure the high quality of MMLU-ProX, we employ a rigorous development process that involves multiple powerful LLMs for translation, followed by expert review to ensure accurate expression, consistent terminology, and cultural relevance.
Building on this, we systematically evaluate 36 state-of-the-art LLMs, including reasoning-enhanced and multilingual-optimized LLMs.
The results reveal significant disparities in the multilingual capabilities of LLMs: While they perform well in high-resource languages, their performance declines markedly in low-resource languages, with gaps of up to 24.3%.
Through MMLU-ProX, we aim to advance the development of more inclusive AI systems and promote equitable access to technology across global contexts.
We plan to continuously expand MMLU-ProX by incorporating additional languages to further enhance its coverage and utility for the global AI research community.`

Homepage: https://mmluprox.github.io

Huggingface:
- https://huggingface.co/datasets/li-lab/MMLU-ProX
- https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite

### Citation

```bibtex
@article{xuan2025mmlu,
  title={Mmlu-prox: A multilingual benchmark for advanced large language model evaluation},
  author={Weihao Xuan and Rui Yang and Heli Qi and Qingcheng Zeng and Yunze Xiao and Aosong Feng and Dairui Liu and Yun Xing and Junjue Wang and Fan Gao and Jinghui Lu and Yuang Jiang and Huitao Li and Xin Li and Kunyu Yu and Ruihai Dong and Shangding Gu and Yuekang Li and Xiaofei Xie and Felix Juefei-Xu and Foutse Khomh and Osamu Yoshie and Qingyu Chen and Douglas Teodoro and Nan Liu and Randy Goebel and Lei Ma and Edison Marrese-Taylor and Shijian Lu and Yusuke Iwasawa and Yutaka Matsuo and Irene Li},
  journal={arXiv preprint arXiv:2503.10497},
  year={2025}
}
```

### Groups and Tasks

#### Groups

* `mmlu_pro_{lang}`: 'All 14 subjects of the mmlu_prox dataset in {lang}, evaluated following the methodology in mmlu_pro's original implementation'
* `mmlu_prox_lite_{lang}`: 'All 14 subjects of the mmlu_prox_lite dataset in {lang}, evaluated following the methodology in mmlu_pro's original implementation'

Available options for `{lang}`:
- af
- ar
- bn
- cs
- de
- en
- es
- fr
- hi
- hu
- id
- it
- ja
- ko
- mr
- ne
- pt
- ru
- sr
- sw
- te
- th
- uk
- ur
- vi
- wo
- yo
- zh
- zu

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


The following tasks evaluate subjects in the mmlu_prox_lite dataset
- `mmlu_prox_lite_{lang}_biology`
- `mmlu_prox_lite_{lang}_business`
- `mmlu_prox_lite_{lang}_chemistry`
- `mmlu_prox_lite_{lang}_computer_science`
- `mmlu_prox_lite_{lang}_economics`
- `mmlu_prox_lite_{lang}_engineering`
- `mmlu_prox_lite_{lang}_health`
- `mmlu_prox_lite_{lang}_history`
- `mmlu_prox_lite_{lang}_law`
- `mmlu_prox_lite_{lang}_math`
- `mmlu_prox_lite_{lang}_other`
- `mmlu_prox_lite_{lang}_philosophy`
- `mmlu_prox_lite_{lang}_physics`
- `mmlu_prox_lite_{lang}_psychology`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
