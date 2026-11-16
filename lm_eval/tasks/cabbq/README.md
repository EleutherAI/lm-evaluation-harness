# Catalan Bias Benchmark for Question Answering (CaBBQ)

### Paper

Title: `EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering`

Abstract: [https://arxiv.org/abs/2507.11216](https://arxiv.org/abs/2507.11216)

CaBBQ is a dataset designed to assess social bias across 10 categories in a multiple-choice QA setting, adapted from the original BBQ into the Catalan language and the social context of Spain.

It is fully parallel with the `esbbq` task group, the version in Spanish.

### Citation

```
@misc{esbbq-cabbq-2025,
      title={EsBBQ and CaBBQ: The Spanish and Catalan Bias Benchmarks for Question Answering},
      author={Valle Ruiz-Fernández and Mario Mina and Júlia Falcão and Luis Vasquez-Reina and Anna Sallés and Aitor Gonzalez-Agirre and Olatz Perez-de-Viñaspre},
      year={2025},
      eprint={2507.11216},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.11216},
}
```

### Groups and Tasks

#### Groups

* `cabbq`: Contains the subtasks that covers all demographic categories.

### Tasks

`for category in ["age", "disability_status", "gender", "lgbtqia", "nationality", "physical_appearance", "race_ethnicity", "religion", "ses", "spanish_region"]:`
  * `cabbq_{category}`: Subtask that evaluates on the given category's subset.

### Metrics

CaBBQ is evaluated with the following 4 metrics, at the level of each subtask and with aggregated values for the entire group:

* `acc_ambig`: Accuracy over ambiguous instances.
* `acc_disambig`: Accuracy over disambiguated instances.
* `bias_score_ambig`: Bias score over ambiguous instances.
* `bias_score_disambig`: Bias score over disambiguated instances.

See the paper for a thorough explanation and the formulas of these metrics.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
