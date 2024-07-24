# TELEIA

### Paper

Title: `Spanish Language Benchmark for Artificial Intelligence Models (TELEIA)`

Abstract: `This dataset is designed to evaluate Large Language Models (LLMs) in Spanish language proficiency. It comprises three distinct components, each targeting different aspects of language competency. The TELEIA_Cervantes_AVE component assesses vocabulary and grammatical structures through fill-in-the-gap questions, mirroring the format of the Cervantes AVE exam, with each entry including a sentence with a gap, four options to fill the gap, and the correct answer. The TELEIA_PCE component focuses on morphology and semantics, featuring short questions and incomplete sentences similar to the PCE exam format, where each entry consists of a question or incomplete sentence, three options, and the correct answer. Lastly, the TELEIA_SIELE component evaluates reading comprehension skills using text-based questions modeled after the SIELE exam, with each entry including a question related to a given text, three answer options, and the correct answer. This dataset provides a robust framework for assessing various dimensions of Spanish language proficiency in LLMs, facilitating the development and benchmarking of AI models in Spanish language understanding and generation tasks.`

Homepage: [TELEIA Dataset](https://zenodo.org/records/12571763)


### Citation

```
@dataset{spanish_benchmark_teleia,
  author       = {Marina-Ruiz, M. and
                  Molinero, Nina and
                  Martín-Gascón, Elena and
                  Galar, Miguel and
                  Fernández, Raquel and
                  Camarero, Javier and
                  Rodríguez, Pedro},
  title        = {{Spanish Language Benchmark for Artificial Intelligence Models (TELEIA)}},
  month        = jan,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.12571763},
  url          = {https://doi.org/10.5281/zenodo.12571763}
}
```

### Groups and Tasks

#### Groups

* *`teleia`*: *`The dataset comprises three components designed to evaluate Spanish language proficiency in Large Language Models (LLMs).`*

#### Tasks

The following tasks evaluate Spanish language proficiency using the TELEIA dataset components:

* `teleia_cervantes_ave`: Evaluates vocabulary and grammatical structures using multiple-choice questions with four options, following the format of the Cervantes AVE exam.

* `teleia_pce`: Assesses morphology and semantics through short questions or sentences to be completed, with three options to choose from, resembling the PCE exam style.

* `teleia_siele`: Tests reading comprehension skills using text-based questions with three answer options, modeled after the SIELE exam's reading comprehension task.

Each task uses loglikelihood-based multiple-choice scoring to evaluate the performance of LLMs on Spanish language proficiency.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [X] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
