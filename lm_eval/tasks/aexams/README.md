# Arabic EXAMS

### Paper

EXAMS: a resource specialized in multilingual high school exam questions.
The original paper [EXAMS](https://aclanthology.org/2020.emnlp-main.438/)

The Arabic EXAMS dataset includes five subjects

  - Islamic studies
  - Biology
  - Physics
  - Science
  - Social

The original dataset [EXAMS-QA](https://github.com/mhardalov/exams-qa)

EXAMS is a benchmark dataset for cross-lingual and multilingual question answering for high school examinations.
With 24,000 high-quality high school exam questions in 16 languages, covering 8 language families and 24 school subjects from Natural Sciences and Social Sciences, among others.
EXAMS offers unique fine-grained evaluation framework across multiple languages and subjects

Homepage for Arabic EXAMS: [EXAMS Arabic Homepage](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval/benchmark_eval/benchmarks/EXAMS_Arabic)

### Citation


### Groups and Tasks

#### Groups

- `EXAMS Arabic`: include IslamicStudies, Biology, Science, Physics, Social.

#### Tasks


The following tasks evaluate subjects in Arabic EXAMS dataset using loglikelihood-based multiple-choice scoring:
- `aexams_IslamicStudies`
- `aexams_Biology`
- `aexams_Science`
- `aexams_Physics`
- `aexams_Social`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * [x] Yes, original implementation contributed by author of the benchmark

If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
