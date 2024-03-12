# ArabicMMLU

### Paper

ArabicMMLU: Measuring massive multitask language understanding in Arabic
This dataset has been translated from the original MMLU with the help of GPT-4.

The original data [MMLU](https://arxiv.org/pdf/2009.03300v3.pdf)

The translation has been done with AceGPT researchers [AceGPT](https://arxiv.org/abs/2309.12053)

ArabicMMLU is a comprehensive evaluation benchmark specifically designed to evaluate the knowledge and reasoning abilities of LLMs within the context of Arabic language and culture.
ArabicMMLU covers a wide range of subjects, comprising 57 topics that span from elementary to advanced professional levels.

Homepage: [AceGPT Homepage](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval/benchmark_eval/benchmarks/MMLUArabic)

### Citation


### Groups and Tasks

#### Groups

- `ammlu`: All 57 subjects of the ArabicMMLU dataset, evaluated following the methodology in MMLU's original implementation.

#### Tasks


The following tasks evaluate subjects in the ArabicMMLU dataset using loglikelihood-based multiple-choice scoring:
- `ammlu_{subject_english}`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
