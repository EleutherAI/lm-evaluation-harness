# EQ-Bench

Title: `EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models`

Abstract: https://arxiv.org/abs/2312.06281

EQ-Bench is a benchmark for language models designed to assess emotional intelligence.

Why emotional intelligence? One reason is that it represents a subset of abilities that are important for the user experience, and which isn't explicitly tested by other benchmarks. Another reason is that it's not trivial to improve scores by fine tuning for the benchmark, which makes it harder to "game" the leaderboard.

EQ-Bench is a little different from traditional psychometric tests. It uses a specific question format, in which the subject has to read a dialogue then rate the intensity of possible emotional responses of one of the characters. Every question is interpretative and assesses the ability to predict the magnitude of the 4 presented emotions. The test is graded without the need for a judge (so there is no length bias). It's cheap to run (only 171 questions), and produces results that correlate strongly with human preference (Arena ELO) and multi-domain benchmarks like MMLU.

Homepage: https://eqbench.com/

### Citation

```bibtex
@misc{paech2023eqbench,
	title={EQ-Bench: An Emotional Intelligence Benchmark for Large Language Models},
	author={Samuel J. Paech},
	year={2023},
	eprint={2312.06281},
	archivePrefix={arXiv},
	primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `eq_bench`

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
