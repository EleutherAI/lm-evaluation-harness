# EgyMMLU

### Paper

Title: NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities

Abstract: [https://arxiv.org/abs/2505.18383](https://arxiv.org/abs/2505.18383)

EgyMMLU is a benchmark designed to evaluate the performance of large language models in Egyptian Arabic. It contains 22,027 multiple-choice questions covering 44 subjects, translated from parts of the Massive Multitask Language Understanding (MMLU) and ArabicMMLU benchmarks. The dataset was translated using `google/gemma-3-27b-it`.

Homepage: [https://huggingface.co/datasets/UBC-NLP/EgyMMLU](https://huggingface.co/datasets/UBC-NLP/EgyMMLU)


### Citation

```
@article{mekki2025nilechatlinguisticallydiverseculturally,
  title={NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities},
  author={Abdellah El Mekki and Houdaifa Atou and Omer Nacar and Shady Shehata and Muhammad Abdul-Mageed},
  year={2025},
  eprint={2505.18383},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.18383},
}
```

### Groups and Tasks

#### Groups

* `egymmlu`: evaluates all EgyMMLU tasks.

#### Tags
Source-based tags:

* `egymmlu_mmlu`: evaluates EgyMMLU tasks that were translated from MMLU.
* `egymmlu_ar_mmlu`: evaluates EgyMMLU tasks that were translated from ArabicMMLU.

Category-based tags:

* `egymmlu_stem`: evaluates EgyMMLU STEM tasks.
* `egymmlu_social_sciences`: evaluates EgyMMLU social sciences tasks.
* `egymmlu_humanities`: evaluates EgyMMLU humanities tasks.
* `egymmlu_language`: evaluates EgyMMLU language tasks.
* `egymmlu_other`: evaluates other EgyMMLU tasks.

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
