# TruthfulQA-Multi

## Paper

Title: `Truth Knows No Language: Evaluating Truthfulness Beyond English`

Abstract: `[https://arxiv.org/abs/2502.09387v1](https://arxiv.org/abs/2502.09387v1)`

We introduce a professionally translated extension of the TruthfulQA benchmark designed to evaluate truthfulness in Basque, Catalan, Galician, and Spanish. Truthfulness evaluations of large language models (LLMs) have primarily been conducted in English. However, the ability of LLMs to maintain truthfulness across languages remains under-explored. Our study evaluates 12 state-of-the-art open LLMs, comparing base and instruction-tuned models using human evaluation, multiple-choice metrics, and LLM-as-a-Judge scoring. Our findings reveal that, while LLMs perform best in English and worst in Basque (the lowest-resourced language), overall truthfulness discrepancies across languages are smaller than anticipated. Furthermore, we show that LLM-as-a-Judge correlates more closely with human judgments than multiple-choice metrics, and that informativeness plays a critical role in truthfulness assessment. Our results also indicate that machine translation provides a viable approach for extending truthfulness benchmarks to additional languages, offering a scalable alternative to professional translation. Finally, we observe that universal knowledge questions are better handled across languages than context- and time-dependent ones, highlighting the need for truthfulness evaluations that account for cultural and temporal variability. Dataset and code are publicly available under open licenses.

### Citation

```text
@misc{figueras2025truthknowslanguageevaluating,
      title={Truth Knows No Language: Evaluating Truthfulness Beyond English},
      author={Blanca Calvo Figueras and Eneko Sagarzazu and Julen Etxaniz and Jeremy Barnes and Pablo Gamallo and Iria De Dios Flores and Rodrigo Agerri},
      year={2025},
      eprint={2502.09387},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.09387},
}
```

### Groups, Tags, and Tasks

#### Groups

* `truthfulqa`: This task follows the [TruthfulQA dataset](https://arxiv.org/abs/2109.07958), but expands it to new languages.

#### Tasks

* `truthfulqa-multi_mc2_es`: `Multiple-choice, multiple answers in Spanish`
* `truthfulqa-multi_gen_es`: `Answer generation in Spanish`
* `truthfulqa-multi_mc2_ca`: `Multiple-choice, multiple answers in Catalan`
* `truthfulqa-multi_gen_ca`: `Answer generation in Catalan`
* `truthfulqa-multi_mc2_eu`: `Multiple-choice, multiple answers in Basque`
* `truthfulqa-multi_gen_eu`: `Answer generation in Basque`
* `truthfulqa-multi_mc2_gl`: `Multiple-choice, multiple answers in Galician`
* `truthfulqa-multi_gen_gl`: `Answer generation in Galician`
* `truthfulqa-multi_mc2_en`: `Multiple-choice, multiple answers in English`
* `truthfulqa-multi_gen_en`: `Answer generation in English`

### Checklist

For adding novel benchmarks/datasets to the library:

* [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [X] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog
