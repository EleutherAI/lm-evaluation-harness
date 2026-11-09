# Task-name

### Paper

Title: [MELA: Multilingual Evaluation of Linguistic Acceptability](https://arxiv.org/abs/2311.09033)

**Abstract**: In this work, we present the largest benchmark to date on linguistic acceptability: Multilingual Evaluation of Linguistic Acceptability -- MELA, with 46K samples covering 10 languages from a diverse set of language families. We establish LLM baselines on this benchmark, and investigate cross-lingual transfer in acceptability judgements with XLM-R. In pursuit of multilingual interpretability, we conduct probing experiments with fine-tuned XLM-R to explore the process of syntax capability acquisition. Our results show that GPT-4o exhibits a strong multilingual ability, outperforming fine-tuned XLM-R, while open-source multilingual models lag behind by a noticeable gap. Cross-lingual transfer experiments show that transfer in acceptability judgment is non-trivial: 500 Icelandic fine-tuning examples lead to 23 MCC performance in a completely unrelated language -- Chinese. Results of our probing experiments indicate that training on MELA improves the performance of XLM-R on syntax-related tasks.

Homepage: https://github.com/sjtu-compling/MELA

### Citation

```
@inproceedings{zhang2023mela,
  author       = {Ziyin Zhang and
                  Yikang Liu and
                  Weifang Huang and
                  Junyu Mao and
                  Rui Wang and
                  Hai Hu},
  title        = {{MELA:} Multilingual Evaluation of Linguistic Acceptability},
  booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2311.09033}
}
```

### Groups and Tasks

#### Groups

- `mela`: multilingual evaluation of linguistic acceptability

#### Tasks

- `mela_en`: English
- `mela_zh`: Chinese
- `mela_it`: Italian
- `mela_ru`: Russian
- `mela_de`: Germany
- `mela_fr`: French
- `mela_es`: Spanish
- `mela_ja`: Japanese
- `mela_ar`: Arabic
- `mela_ar`: Icelandic

### Checklist

For adding novel benchmarks/datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
