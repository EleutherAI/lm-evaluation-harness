# BertaQA

### Paper

Title: BertaQA: How Much Do Language Models Know About Local Culture?

Abstract: https://arxiv.org/abs/2406.07302

Large Language Models (LLMs) exhibit extensive knowledge about the world, but most evaluations have been limited to global or anglocentric subjects. This raises the question of how well these models perform on topics relevant to other cultures, whose presence on the web is not that prominent. To address this gap, we introduce BertaQA, a multiple-choice trivia dataset that is parallel in English and Basque. The dataset consists of a local subset with questions pertinent to the Basque culture, and a global subset with questions of broader interest. We find that state-of-the-art LLMs struggle with local cultural knowledge, even as they excel on global topics. However, we show that continued pre-training in Basque significantly improves the models' performance on Basque culture, even when queried in English. To our knowledge, this is the first solid evidence of knowledge transfer from a low-resource to a high-resource language. Our analysis sheds light on the complex interplay between language and knowledge, and reveals that some prior findings do not fully hold when reassessed on local topics. Our dataset and evaluation code are available under open licenses at https://github.com/juletx/BertaQA.

Homepage: https://github.com/juletx/BertaQA

### Citation

```
@misc{etxaniz2024bertaqa,
      title={BertaQA: How Much Do Language Models Know About Local Culture?},
      author={Julen Etxaniz and Gorka Azkune and Aitor Soroa and Oier Lopez de Lacalle and Mikel Artetxe},
      year={2024},
      eprint={2406.07302},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

- `bertaqa`: Group of BertaQA tasks.

#### Tasks

- `bertaqa_eu`: Trivia questions in Basque.
- `bertaqa_en`: Trivia questions in English, human-translated from Basque.
- `bertaqa_en_mt_*`: Trivia questions in English, machine-translated from Basque with different models.

### Checklist

For adding novel benchmarks/datasets to the library:

- [ ] Is the task an existing benchmark in the literature?
  - [ ] Have you referenced the original paper that introduced the task?
  - [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
