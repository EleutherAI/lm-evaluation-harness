# Japanese Leaderboard

### Paper

Title: `paper titles goes here`

Abstract: `link to paper PDF or arXiv abstract goes here`

`Short description of paper / benchmark goes here:`

Homepage: `homepage to the benchmark's website goes here, if applicable`


### Citation

```
BibTeX-formatted citation goes here
```

### Groups, Tags, and Tasks

#### Tags

* `japanese_leaderboard`: runs all tasks defined in this directory
* `ja_leaderboard`: alias for `japanese_leaderboard`

#### Tasks

* `ja_leaderboard_jaqket_v2`:
* `ja_leaderboard_jcommonsenseqa`:
* `ja_leaderboard_jnli`:
* `ja_leaderboard_jsquad`:
* `ja_leaderboard_marc_ja`:
* `ja_leaderboard_mgsm`:
* `ja_leaderboard_xlsum`:
* `ja_leaderboard_xwinograd`:

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
