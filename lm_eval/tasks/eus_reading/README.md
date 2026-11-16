# EusReading

### Paper

Title: Latxa: An Open Language Model and Evaluation Suite for Basque

Abstract: https://arxiv.org/abs/2403.20266

EusReading consists of 352 reading comprehension exercises (irakurmena) sourced from the set of past EGA exams from 1998 to 2008. Each test generally has 10 multiple-choice questions, with 4 choices and a single correct answer. These exercises are more challenging than Belebele due to the complexity and length of the input texts. As a result, EusReading is useful to measure long context understanding of models.

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

There are no groups.

#### Tasks

* `eus_reading`: EusReading consists of 352 reading comprehension exercises (irakurmena) sourced from the set of past EGA exams from 1998 to 2008.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
