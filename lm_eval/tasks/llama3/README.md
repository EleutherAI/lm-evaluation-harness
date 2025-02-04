# Task-name

### Paper

Title: LLAMA Evals

Abstract: Evals reproducing those provided by the LLAMA team in the Hugging Face repo.

`Short description of paper / benchmark goes here:`

Homepage: `https://huggingface.co/collections/meta-llama/llama-31-evals-66a2c5a14c2093e58298ac7f`

Note: The tasks are formatted to be run with apply_chat_template and fewshot_as_multiturn.
### Citation

```
BibTeX-formatted citation goes here
```

### Groups, Tags, and Tasks

#### Groups

* `group_name`: `Short description`

#### Tags

* `tag_name`: `Short description`

#### Tasks

* `mmlu_llama`: `generation variant of MMLU`
* `arc_chalenge_chat`: `generation variant of ARC-Challenge using MMLU format`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
