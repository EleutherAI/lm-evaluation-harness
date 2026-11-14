# Sindhi-Reasoning

## Paper

Title: `Enabling Structured Reasoning in Sindhi with Culturally Grounded Instruction Tuning`

Paper: https://aclanthology.org/2026.loreslm-1.22/

Sindhi-Reasoning-Instruct is a culturally grounded Sindhi instruction dataset covering deductive,
inductive, causal and commonsense reasoning. The test set has 100 instruction-input-answer items.

Homepage: https://github.com/KamyarZeinalipour/Sindhi-Reasoning

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/sindi_reasoning

### Citation

```text
@inproceedings{mehak-etal-2026-enabling,
    title = "Enabling Structured Reasoning in {S}indhi with Culturally Grounded Instruction Tuning",
    author = "Mehak, Mehak and Zeinalipour, Kamyar and Soomro, Pireh and Chesi, Cristiano and Gori, Marco and Maggini, Marco",
    booktitle = "Proceedings of the Second Workshop on Language Models for Low-Resource Languages ({L}o{R}es{LM} 2026)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.loreslm-1.22/",
    doi = "10.18653/v1/2026.loreslm-1.22",
    pages = "239--258"
}
```

### Groups, Tags, and Tasks

#### Tasks

* `sindhi_reasoning`: the model answers a Sindhi reasoning instruction; chrF against the reference answer.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

This task reports chrF only. The paper's own evaluation scripts are in the GitHub repository;
numbers from the two setups are not directly comparable.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
