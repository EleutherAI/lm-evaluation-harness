# HAE-RAE BENCH

### Paper

Title: `HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models`

Abstract: `Large Language Models (LLMs) trained on massive corpora demonstrate impressive capabilities in a wide range of tasks. While there are ongoing efforts to adapt these models to languages beyond English, the attention given to their evaluation methodologies remains limited. Current multilingual benchmarks often rely on back translations or re-implementations of English tests, limiting their capacity to capture unique cultural and linguistic nuances. To bridge this gap for the Korean language, we introduce HAE-RAE Bench, a dataset curated to challenge models lacking Korean cultural and contextual depth. The dataset encompasses six downstream tasks across four domains: vocabulary, history, general knowledge, and reading comprehension. Contrary to traditional evaluation suites focused on token or sequence classification and specific mathematical or logical reasoning, HAE-RAE Bench emphasizes a model's aptitude for recalling Korean-specific knowledge and cultural contexts. Comparative analysis with prior Korean benchmarks indicates that the HAE-RAE Bench presents a greater challenge to non-native models, by disturbing abilities and knowledge learned from English being transferred.`

Homepage: https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH

### Citation

@misc{son2023haerae,
      title={HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models},
      author={Guijin Son and Hanwool Lee and Suwan Kim and Huiseo Kim and Jaecheol Lee and Je Won Yeom and Jihyu Jung and Jung Woo Kim and Songseong Kim},
      year={2023},
      eprint={2309.02706},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

### Groups and Tasks

#### Groups

* `haerae`: 'It consists of five tasks provided in the HAERAE-BENCH paper. 'Reading Comprehension' was excluded from the implementation due to copyright issues. We will include it in the next haerae update. For other tasks, some part of data may be replaced or increased with the production of Haerae v1.1. Please note this when using it.'

#### Tasks

The following tasks evaluate subjects in the HaeRae dataset

- `haerae_standard_nomenclature`
- `haerae_loan_word`
- `haerae_rare_word`
- `haerae_general_knowledge`
- `haerae_history`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
