# Educational Crosswords

## Papers

This folder collects tasks from a line of work on generating and solving educational crossword
clues with LLMs:

* `Clue-Instruct: Text-Based Clue Generation for Educational Crossword Puzzles` (LREC-COLING 2024): https://aclanthology.org/2024.lrec-main.297/
* `Harnessing LLMs for Educational Content-Driven Italian Crossword Generation` (CLiC-it 2024): https://aclanthology.org/2024.clicit-1.110/
* `From Arabic Text to Puzzles: LLM-Driven Development of Arabic Educational Crosswords` (LoResLM 2025): https://aclanthology.org/2025.loreslm-1.36/
* `Surprisal and Crossword Clues Difficulty: Evaluating Linguistic Processing between LLMs and Humans` (CLiC-it 2025): https://aclanthology.org/2025.clicit-1.50/

In the generation tasks the model reads a Wikipedia passage and a target answer and writes an
educational crossword clue for it. The solving task asks the model to solve 159 Italian clues whose
difficulty was rated by human solvers.

Data: [clue-instruct](https://huggingface.co/datasets/azugarini/clue-instruct),
[ita_cw_text](https://huggingface.co/datasets/Kamyar-zeinalipour/ita_cw_text),
[Arabic-Clue-Instruct](https://huggingface.co/datasets/Kamyar-zeinalipour/Arabic-Clue-Instruct),
[Crossword-Surprisal-IT](https://huggingface.co/datasets/Kamyar-zeinalipour/Crossword-Surprisal-IT)

### Citation

```text
@inproceedings{zugarini-etal-2024-clue,
    title = "Clue-Instruct: Text-Based Clue Generation for Educational Crossword Puzzles",
    author = "Zugarini, Andrea and Zeinalipour, Kamyar and Kadali, Surya Sai and Maggini, Marco and Gori, Marco and Rigutini, Leonardo",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation ({LREC}-{COLING} 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.297/",
    pages = "3347--3356"
}

@inproceedings{zeinalipour-etal-2024-harnessing,
    title = "Harnessing {LLM}s for Educational Content-Driven {I}talian Crossword Generation",
    author = "Zeinalipour, Kamyar and Fusco, Achille and Zanollo, Asya and Maggini, Marco and Gori, Marco",
    booktitle = "Proceedings of the Tenth {I}talian Conference on Computational Linguistics ({CL}i{C}-it 2024)",
    month = dec,
    year = "2024",
    address = "Pisa, Italy",
    publisher = "CEUR Workshop Proceedings",
    url = "https://aclanthology.org/2024.clicit-1.110/",
    pages = "1013--1023"
}

@inproceedings{zeinalipour-etal-2025-arabic,
    title = "From {A}rabic Text to Puzzles: {LLM}-Driven Development of {A}rabic Educational Crosswords",
    author = "Zeinalipour, Kamyar and Saad, Moahmmad and Maggini, Marco and Gori, Marco",
    booktitle = "Proceedings of the First Workshop on Language Models for Low-Resource Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.loreslm-1.36/",
    pages = "479--495"
}

@inproceedings{iaquinta-etal-2025-surprisal,
    title = "Surprisal and Crossword Clues Difficulty: Evaluating Linguistic Processing between {LLM}s and Humans",
    author = "Iaquinta, Tommaso and Zanollo, Asya and Fusco, Achille and Zeinalipour, Kamyar and Chesi, Cristiano",
    booktitle = "Proceedings of the Eleventh {I}talian Conference on Computational Linguistics ({CL}i{C}-it 2025)",
    month = sep,
    year = "2025",
    address = "Cagliari, Italy",
    publisher = "CEUR Workshop Proceedings",
    url = "https://aclanthology.org/2025.clicit-1.50/",
    pages = "512--525"
}
```

### Groups, Tags, and Tasks

#### Groups

* `edu_crosswords`: the three clue-generation tasks.

#### Tasks

* `clue_instruct_en`: English clue generation (600 test items); sentence chrF against the closest of the reference clues.
* `italian_clue_generation`: Italian clue generation (600 test items); sentence chrF against the reference clue.
* `arabic_clue_generation`: Arabic clue generation (497 test items); sentence chrF against the closest of the reference clues.
* `italian_crossword_solving`: solve 159 Italian clues given the answer length; exact match on the first word of the output, ignoring case and accents.

chrF measures closeness to the reference clues only; it does not check that a clue is correct or
that it avoids giving the answer away.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The papers evaluate generated clues with human ratings and other measures; these tasks use chrF,
so the numbers are not directly comparable. The Surprisal paper studies clue difficulty rather than
solving accuracy; `italian_crossword_solving` reuses its clue set as a solving test.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
