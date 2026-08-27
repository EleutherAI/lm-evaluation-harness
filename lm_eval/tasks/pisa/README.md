# PisaBench

### Paper

Title: PISA-Bench: The PISA Index as a Multilingual and Multimodal Metric for the Evaluation of Vision-Language Models

Abstract: https://arxiv.org/abs/2510.24792

Vision-language models (VLMs) have demonstrated remarkable progress in multimodal reasoning. However, existing benchmarks remain limited in terms of high-quality, human-verified examples. Many current datasets rely on synthetically generated content by large language models (LLMs). Furthermore, most datasets are limited to English, as manual quality assurance of translated samples is time-consuming and costly. To fill this gap, we introduce PISA-Bench, a multilingual benchmark derived from English examples of the expert-created PISA tests, a unified framework for the assessment of student competencies in over eighty countries. Each example consists of human-extracted instructions, questions, answer options, and images, enriched with question type categories, and has been translated from English into five additional languages (Spanish, German, Chinese, French, and Italian), resulting in a fully parallel corpus covering six languages. We evaluate state-of-the-art vision-language models on PISA-Bench and find that especially small models (<20B parameters) fail to achieve high test scores. We further find substantial performance degradation on non-English splits as well as high error-rates when models are tasked with spatial and geometric reasoning. By releasing the dataset and evaluation framework, we provide a resource for advancing research on multilingual multimodal reasoning.

HuggingFace Dataset: https://huggingface.co/datasets/PisaBench/pisa-bench

### Citation

```@misc{haller2025pisabenchpisaindexmultilingual,
      title={PISA-Bench: The PISA Index as a Multilingual and Multimodal Metric for the Evaluation of Vision-Language Models},
      author={Patrick Haller and Fabio Barth and Jonas Golde and Georg Rehm and Alan Akbik},
      year={2025},
      eprint={2510.24792},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.24792},
}```

### Groups, Tags, and Tasks

#### Groups

* `pisa`: Evaluates over all language splits with substring matching for answer evaluation.
* `pisa_llm_judged`: Evaluates over all language splits with LLM-based answer evaluation (requires OpenAI api key).

#### Tags

None.

#### Tasks

* `pisa_en`
* `pisa_de`
* `pisa_es`
* `pisa_fr`
* `pisa_it`
* `pisa_ch`
* `pisa_en_llm_judged`
* `pisa_de_llm_judged`
* `pisa_es_llm_judged`
* `pisa_fr_llm_judged`
* `pisa_it_llm_judged`
* `pisa_ch_llm_judged`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
