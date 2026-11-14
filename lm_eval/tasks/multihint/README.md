# MultiHint-9K

## Paper

Title: `MultiHint-9K: Orchestrating Human and AI Agency for Scalable Pedagogical Hint Generation`

Paper: https://ceur-ws.org/Vol-4248/short11.pdf

MultiHint-9K has 9,987 verified passage-question-answer-hint tuples in English, Italian and Farsi,
built by a pipeline in which human annotators set quality rubrics and AI agents generate, critique
and refine Socratic hints. The test set has 300 items (100 English, 113 Italian, 87 Farsi).

Homepage: https://github.com/KamyarZeinalipour/MultiHint-9K

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/MultiHint-9K

### Citation

```text
@inproceedings{zeinalipour2026multihintk,
  title = {MultiHint-9K: Orchestrating Human and AI Agency for Scalable Pedagogical Hint Generation},
  author = {Kamyar Zeinalipour and Amir Sadeghi and Giovanni Angelini and Leonardo Rigutini and Marco Gori and Marco Maggini},
  booktitle = {Proceedings of the Workshop on Orchestrating Human and AI Agency for Proactive and Reflective Learning (HAI-Agency 2026), co-located with AIED 2026},
  year = {2026},
  series = {CEUR Workshop Proceedings},
  volume = {4248},
  url = {https://ceur-ws.org/Vol-4248/short11.pdf}
}
```

### Groups, Tags, and Tasks

#### Groups

* `multihint`: the three languages.

#### Tasks

* `multihint_en`: given a passage, a question and its answer, write a hint that leads to the answer without revealing it (English); chrF against the reference hint.
* `multihint_it`: the same in Italian.
* `multihint_fa`: the same in Farsi.

chrF measures closeness to the reference hint only; it does not check that the hint avoids giving
the answer away.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
