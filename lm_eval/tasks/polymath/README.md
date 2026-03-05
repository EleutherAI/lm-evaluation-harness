# PolyMath

### Paper

PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts

PolyMath is a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels, with 9,000 high-quality problem samples. The benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs.

Homepage: https://arxiv.org/abs/2504.18428

GitHub: https://github.com/QwenLM/PolyMath

Hugging Face: https://huggingface.co/datasets/Qwen/PolyMath

### Citation

```
@article{wang2025polymath,
title={PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts},
author={Yiming Wang and Pei Zhang and Jialong Tang and Haoran Wei and Baosong Yang and Rui Wang and Chenshu Sun and Feitong Sun and Jiran Zhang and Junxuan Wu and Qiqian Cang and Yichang Zhang and Fei Huang and Junyang Lin and Fei Huang and Jingren Zhou},
journal={arXiv preprint arXiv:2504.18428},
year={2025},
primaryClass={cs.CL},
url={https://arxiv.org/abs/2504.18428}, 
}
```

### Groups and Tasks

PolyMath tasks follow a two-dimensional structure: **language** × **difficulty**.

There are 18 supported languages: `ar`, `bn`, `de`, `en`, `es`, `fr`, `id`, `it`, `ja`, `ko`, `ms`, `pt`, `ru`, `sw`, `te`, `th`, `vi`, `zh`.

| Language code | Language name |
|----------------|---------------|
| `ar` | Arabic |
| `bn` | Bengali |
| `de` | German |
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `id` | Indonesian |
| `it` | Italian |
| `ja` | Japanese |
| `ko` | Korean |
| `ms` | Malay |
| `pt` | Portuguese |
| `ru` | Russian |
| `sw` | Swahili |
| `te` | Telugu |
| `th` | Thai |
| `vi` | Vietnamese |
| `zh` | Chinese |

There are 4 difficulty tiers: `low`, `medium`, `high`, `top`.

| Difficulty tier | Description |
|---|---|
| `low` | Arithmetic problems from MGSM and P-MMeval |
| `medium` | Undergraduate exercises, entrance exams (Gaokao, postgraduate), entry-level competitions (AMC, provincial CNMO) |
| `high` | Mid-tier national competitions (AIME, national CNMO) |
| `top` | International/national Olympiads (IMO, Putnam, USAMO) and HLE frontier math |

#### Groups

| Group | Description | Number of tasks |
|---|---|---|
| `polymath` | All languages (18) × All difficulty tiers (4) | 72 |
| `polymath_<lang>` | All difficulty tiers (4) for one language, e.g. `polymath_fr` | 4 |
| `polymath_<difficulty>` | All languages (18) at one difficulty tier, e.g. `polymath_high` | 18 |

#### Tasks

Each of the 72 individual task contains a total of 125 math problems and can be run using the naming pattern `polymath_<lang>_<difficulty>` (e.g., `polymath_vi_top`).

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
* [x] Have you referenced the original paper that introduced the task?
* [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
* [ ] Checked for equivalence with v0.3.0 LM Evaluation Harness

