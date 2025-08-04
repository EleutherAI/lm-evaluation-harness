# Task-name

### Paper

Title: `Are We Donewith MMLU?`

Abstract: `https://arxiv.org/pdf/2406.04127`

`The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.`

Homepage: `https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0`

### Citation

```
BibTeX
@misc{edinburgh2024mmlu,
      title={Are We Done with MMLU?},
      author={Aryo Pradipta Gema and Joshua Ong Jun Leang and Giwon Hong and Alessio Devoto and
      Alberto Carlo Maria Mancino and Rohit Saxena and Xuanli He and Yu Zhao and Xiaotang Du and
      MohammadRezaGhasemi Madani and Claire Barale and Robert McHardy and Joshua Harris and
      Jean Kaddour and Emile van Krieken and Pasquale Minervini},
      year={2025},
      eprint={2406.04127},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups, Tags, and Tasks

#### Groups

- `stem`
- `other`
- `social sciences`
- `humanities`

#### Tasks

- `mmlu_stem_generative`
- `mmlu_other_generative`
- `mmlu_social_sciences_generative`
- `mmlu_humanities_generative`

### Checklist

For adding novel benchmarks/datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [ ] Is the "Main" variant of this task clearly denoted?
- [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

ver 1: PR #2705
First implementation
