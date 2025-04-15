# k_mmlu

### Paper

Title: `KMMLU : Measuring Massive Multitask Language Understanding in Korean`

Abstract: `We propose KMMLU, a new Korean benchmark with 35,030 expert-level multiple-choice questions across 45 subjects ranging from humanities to STEM. Unlike previous Korean benchmarks that are translated from existing English benchmarks, KMMLU is collected from original Korean exams, capturing linguistic and cultural aspects of the Korean language. We test 26 publicly available and proprietary LLMs, identifying significant room for improvement. The best publicly available model achieves 50.54% on KMMLU, far below the average human performance of 62.6%. This model was primarily trained for English and Chinese, not Korean. Current LLMs tailored to Korean, such as Polyglot-Ko, perform far worse. Surprisingly, even the most capable proprietary LLMs, e.g., GPT-4 and HyperCLOVA X, achieve 59.95% and 53.40%, respectively. This suggests that further work is needed to improve Korean LLMs, and KMMLU offers the right tool to track this progress. We make our dataset publicly available on the Hugging Face Hub and integrate the benchmark into EleutherAI's Language Model Evaluation Harness.`

Note: lm-eval-harness is using the micro average as the default. To replicate the test results in the paper, take the macro average for the scores evaluated with lm-eval-harness

Homepage: https://huggingface.co/datasets/HAERAE-HUB/KMMLU

### Citation

@article{son2024kmmlu,
      title={KMMLU: Measuring Massive Multitask Language Understanding in Korean},
      author={Guijin Son and Hanwool Lee and Sungdong Kim and Seungone Kim and Niklas Muennighoff and Taekyoon Choi and Cheonbok Park and Kang Min Yoo and Stella Biderman},
      journal={arXiv preprint arXiv:2402.11548},
      year={2024}
}

### Groups and Tasks

#### Groups

* `kmmlu`: 'All 45 subjects of the KMMLU dataset, evaluated following the methodology in MMLU's original implementation'
* `kmmlu_direct`: 'kmmlu_direct solves questions using a straightforward *generative* multiple-choice question-answering approach'
* `kmmlu_hard`: 'kmmlu_hard comprises difficult questions that at least one proprietary model failed to answer correctly using log-likelihood approach'
* `kmmlu_hard_direct`:  'kmmlu_hard_direct solves questions of kmmlu_hard using direct(generative) approach'
* `kmmlu_hard_cot`: 'kmmlu_hard_cot includes 5-shot of exemplars for chain-of-thought approach'

#### Tasks

The following tasks evaluate subjects in the KMMLU dataset
- `kmmlu_{subject_english}`
- `kmmlu_direct_{subject_english}`

The following tasks evaluate subjects in the KMMLU-Hard dataset
- `kmmlu_hard_{subject_english}`
- `kmmlu_hard_cot_{subject_english}`
- `kmmlu_hard_direct_{subject_english}`


### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
