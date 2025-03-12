# MastermindEval

### Paper

Title: MastermindEval: A Simple But Scalable Reasoning Benchmark

Abstract: https://arxiv.org/abs/2503.05891

In Mastermind, the player has to deduce a hidden sequence of symbols by iteratively
guessing using the feedback provided by the game master. MastermindEval contains pre-played
games of the board game Mastermind using Knuth's algorithm. Each game is pre-played
until only one possible, valid solutions remains. The task is to derive the hidden
sequence of symbol by combining information provided in the prompt. We offer different
splits of varying difficulty: 24 (code length 2, 4 possible colors), 35 (code length 3,
5 possible colors) and 46 (code length 4, 6 possible colors). Each split comes in two
variants - easy and hard -  containing either random codes as wrong answer options
or codes that are very close (only one symbol is changed) compared to the correct code.
We further offer an agentic evaluation in which the LLM plays the game from scratch here.

GitHub repository: https://github.com/flairNLP/mastermind


### Citation

```
@inproceedings{
  golde2025mastermindeval,
  title={MastermindEval: A Simple But Scalable Reasoning Benchmark},
  author={Jonas Golde and Patrick Haller and Fabio Barth and Alan Akbik},
  booktitle={Workshop on Reasoning and Planning for Large Language Models},
  year={2025},
  url={https://openreview.net/forum?id=H4donosutm}
}
```

### Groups, Tags, and Tasks

#### Groups

None.

#### Tags

* `mastermind`: Evaluates all settings.
* `mastermind_easy`: Evaluates all easy settings (random wrong answer options).
* `mastermind_hard`: Evaluates all hard settings (wrong answer options differ in one symbol from the secret code).

#### Tasks

* `mastermind_24_easy`
* `mastermind_24_hard`
* `mastermind_35_easy`
* `mastermind_35_hard`
* `mastermind_46_easy`
* `mastermind_46_hard`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
