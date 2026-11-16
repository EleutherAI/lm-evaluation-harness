# EusTrivia

### Paper

Title: Latxa: An Open Language Model and Evaluation Suite for Basque

Abstract: https://arxiv.org/abs/2403.20266

EusTrivia consists of 1,715 trivia questions from multiple online sources. 56.3\% of the questions are elementary level (grades 3-6), while the rest are considered challenging. A significant portion of the questions focus specifically on the Basque Country, its language and culture. Each multiple-choice question contains two, three or four choices (3.84 on average) and a single correct answer. Five areas of knowledge are covered:

- **Humanities and Natural Sciences** (27.8%): This category encompasses questions about history, geography, biology, ecology and other social and natural sciences.
- **Leisure and Art** (24.5%): This category includes questions on sports and athletes, performative and plastic arts and artists, architecture, cultural events, and related topics.
- **Music** (16.0%): Here are grouped all the questions about music and musicians, both classical and contemporary.
- **Language and Literature** (17.1%): This category is concerned with all kinds of literature productions and writers, as well as metalinguistic questions (e.g., definitions, synonyms, and word usage).
- **Mathematics and ICT** (14.5%): This category covers mathematical problems and questions about ICT, as well as questions about people known for their contributions to these fields of knowledge.

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

* `eus_trivia`: EusTrivia consists of 1,715 trivia questions from multiple online sources.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
