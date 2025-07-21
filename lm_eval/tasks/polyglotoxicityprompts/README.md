# Polyglot Toxicity Prompts

## Paper

Title: `PolygloToxicityPrompts: Multilingual Evaluation of Neural Toxic Degeneration in Large Language Models`

Abstract: https://arxiv.org/abs/2405.09373

Homepage: https://huggingface.co/datasets/ToxicityPrompts/PolygloToxicityPrompts

A multilingual toxicity evaluation benchmark curated from web text. The dataset has 3 splits: full, small, and wildchat containing 25K, 5K and 1K prompts per language respectively.

### Citation

```text
@misc{jain2024polyglotoxicityprompts,
      title={PolygloToxicityPrompts: Multilingual Evaluation of Neural Toxic Degeneration in Large Language Models}, 
      author={Devansh Jain and Priyanshu Kumar and Samuel Gehman and Xuhui Zhou and Thomas Hartvigsen and Maarten Sap},
      year={2024},
      eprint={2405.09373},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Implementation

The implementation is based on the `realtoxicityprompts` task. To run these tasks, you need to set the `PERSPECTIVE_API_KEY` environment variable.

### Groups, Tags, and Tasks

#### Groups

* `polyglotoxicityprompts_full`: `Full split of 25K toxicity prompts per language`
* `polyglotoxicityprompts_small`: `Small split of 5K toxicity prompts per language`
* `polyglotoxicityprompts_wildchat`: `WildChat split of 1K toxicity prompts per language`

#### Tasks

For each group, there are 17 tasks, one for each language.

For the full split, the tasks are:
  - `polyglotoxicityprompts_full_arabic`
  - `polyglotoxicityprompts_full_czech`
  - `polyglotoxicityprompts_full_german`
  - `polyglotoxicityprompts_full_english`
  - `polyglotoxicityprompts_full_spanish`
  - `polyglotoxicityprompts_full_french`
  - `polyglotoxicityprompts_full_hindi`
  - `polyglotoxicityprompts_full_indonesian`
  - `polyglotoxicityprompts_full_italian`
  - `polyglotoxicityprompts_full_japanese`
  - `polyglotoxicityprompts_full_korean`
  - `polyglotoxicityprompts_full_dutch`
  - `polyglotoxicityprompts_full_polish`
  - `polyglotoxicityprompts_full_portuguese`
  - `polyglotoxicityprompts_full_russian`
  - `polyglotoxicityprompts_full_swedish`
  - `polyglotoxicityprompts_full_chinese`

For the small split, the tasks are `polyglotoxicityprompts_small_<language>`.

For the wildchat split, the tasks are `polyglotoxicityprompts_wildchat_<language>`.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
