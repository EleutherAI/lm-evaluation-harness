# PersianMMLU

### Paper

Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?
https://arxiv.org/abs/2404.06644

The Khayyam Challenge (PersianMMLU) is a comprehensive benchmark designed to evaluate the performance of Large Language Models (LLMs) in understanding and processing the
Persian language. This dataset includes 20,192 multiple-choice questions across 38 diverse tasks sourced from Persian educational examinations.

Homepage: https://huggingface.co/spaces/raia-center/PersianMMLU

### Citation

```bibtex

@article{ghahroodi2024khayyam,
  title={Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?},
  author={Ghahroodi, Omid and Nouri, Marzia and Sanian, Mohammad Vali and Sahebi, Alireza and Dastgheib, Doratossadat and Asgari, Ehsaneddin and Baghshah, Mahdieh Soleymani and Rohban, Mohammad Hossein},
  journal={arXiv preprint arXiv:2404.06644},
  year={2024}
}

```

### Groups and Tasks

#### Groups

- `persianmmlu`: We summarized and consolidated all of 38 tasks arrived in paper, which include questions from various fields of study and educational levels, into 21 academic
  disciplines. Then, we categorized these 21 disciplines into 6 types of knowledge (our Tasks). The reason for this was to avoid creating too many insignificant categories with overly
  fine-grained classification, and to prevent an overly general classification that would result in a lack of insightful evaluation.

#### Tasks

- `persianmmlu_humanities`: These include subjects such as
    - economics (`persianmmlu_economy`),
    - geography (`persianmmlu_geography`),
    - history (`persianmmlu_history`),
    - logic (`persianmmlu_logic`),
    - philosophy (`persianmmlu_philosophy`),
    - psychology (`persianmmlu_psychology`),
    - social studies (`persianmmlu_social_studies`),
    - and sociology (`persianmmlu_sociology`).


- `persianmmlu_math_sciences`: These subjects tasks such as
    - discrete math (`persianmmlu_chemistry`)
    - geometry (`persianmmlu_geometry`)
    - math (`persianmmlu_math`)
    - statistics and probability (`persianmmlu_statistics_and_probability`)


- `persianmmlu_natural_sciences`: These subjects tasks such as
    - biology (`persianmmlu_biology`)
    - chemistry (`persianmmlu_chemistry`)
    - geology (`persianmmlu_geology`)
    - physics (`persianmmlu_physics`)
    - science (`persianmmlu_science`)


- `persianmmlu_persian_lit`: These include subjects such as
    - persian literature (`persianmmlu_persian_literature`)


- `persianmmlu_islamic_life`: These include subjects such as
    - islamic life (`persianmmlu_islamic_life`)


- `persianmmlu_intelligence`: These include subjects such as
    - analytics (`persianmmlu_analytics`)
    - intelligence (`persianmmlu_intelligence`)

### Checklist

* [x] Is the task an existing benchmark in the literature?
    * [x] Have you referenced the original paper that introduced the task?
    * [x] If yes, does the original paper provide a reference implementation?
        * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?