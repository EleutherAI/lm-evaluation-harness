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
* `mmlu_pro_llama`: `generation variant of MMLU-PRO`
* `mmlu_cot_llama`: `Chain-of-thought variant of MMLU`
* `mmlu_it_llama`: `Italian version of generation MMLU`
* `mmlu_fr_llama`: `French version of generation MMLU`
* `mmlu_pt_llama`: `Portuguese version of generation MMLU`
* `mmlu_th_llama`: `Thai version of generation MMLU`
* `mmlu_hi_llama`: `Hindi version of generation MMLU`
* `mmlu_es_llama`: `Spanish version of generation MMLU`
* `mmlu_de_llama`: `German version of generation MMLU`
* `arc_challenge_llama`: `generation variant of ARC-Challenge following Meta pre-processing`
* `gsm8k_llama`: `Chain-of-though variant of GSM8k`


**Notes regarding arc_challenge_llama:**

- The original ARC-Challenge dataset contains 8 samples with less than 4 options. Meta filtered these samples out, and `arc_challenge_llama` does the same.
- A small number of samples use 1, 2, 3, 4 as labels. These are replaced by A, B, C, D like the rest in the doc preprocessing.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
