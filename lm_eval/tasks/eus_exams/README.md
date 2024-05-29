# EusExams

### Paper

Title: Latxa: An Open Language Model and Evaluation Suite for Basque

Abstract: https://arxiv.org/abs/2403.20266

EusExams is a collection of tests designed to prepare individuals for Public Service examinations conducted by several Basque institutions, including the public health system Osakidetza, the Basque Government, the City Councils of Bilbao and Gasteiz, and the University of the Basque Country (UPV/EHU). Within each of these groups, there are different exams for public positions, such as administrative and assistant roles. Each multiple-choice question contains 2 to 4 choices (3.90 on average) and one correct answer. The dataset is mostly parallel with 16k questions in Basque and 18k in Spanish.

Homepage: https://github.com/hitz-zentroa/latxa


### Citation

```
@misc{etxaniz2024latxa,
      title={Latxa: An Open Language Model and Evaluation Suite for Basque},
      author={Julen Etxaniz and Oscar Sainz and Naiara Perez and Itziar Aldabe and German Rigau and Eneko Agirre and Aitor Ormazabal and Mikel Artetxe and Aitor Soroa},
      year={2024},
      eprint={2403.20266},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `eus_exams_eu`: The Basque version of the exams.
* `eus_exams_es`: The Spanish version of the exams.

#### Tasks

Basque and Spanish versions of the exams are available as separate tasks starting with `eus_exams_eu` and `eus_exams_es` respectively.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
