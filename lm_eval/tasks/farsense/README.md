# FarSense

## Paper

Title: `FarSense: A Comprehensive Commonsense Benchmark and Evaluation Framework for the Farsi Language`

Paper: https://aclanthology.org/2025.ijcnlp-long.187/

FarSense is a commonsense reasoning benchmark for Farsi. Each item is a short Farsi scenario
written from a Persian Wikipedia passage, followed by one of seven tasks: true/false (two variants),
multiple choice, cloze completion, cause-to-effect, counterfactual reasoning and "why" explanation.
The test set has about 210 items per task.

Homepage: https://github.com/KamyarZeinalipour/FarSense

Data: https://huggingface.co/datasets/Kamyar-zeinalipour/CS_Farsi (the `eval` config holds the
items as plain context, question, options and answer fields).

### Citation

```text
@inproceedings{zeinalipour-etal-2025-farsense,
    title = "{F}ar{S}ense: A Comprehensive Commonsense Benchmark and Evaluation Framework for the {F}arsi Language",
    author = "Zeinalipour, Kamyar and Jamshidi, Neda and Hejazi, Seyedehbahareh and Maggini, Marco and
      Bianchini, Monica and Paoletti, Simone and Gori, Marco",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the {A}sia-Pacific Chapter of the {A}ssociation for {C}omputational {L}inguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.187/",
    doi = "10.18653/v1/2025.ijcnlp-long.187",
    pages = "3529--3599"
}
```

### Groups, Tags, and Tasks

#### Groups

* `farsense`: the two log-likelihood tasks (multiple choice and true/false), accuracy averaged over items.
* `farsense_generation`: the four generative tasks.

#### Tasks

* `farsense_multiple_choice`: four options; the model's log-likelihood of each option text picks the answer (accuracy).
* `farsense_true_false`: true/false statements about the scenario, both variants pooled (accuracy).
* `farsense_cloze`: the model writes the missing phrase; normalized exact match.
* `farsense_cause_effect`: the model writes the effect of a given cause; chrF.
* `farsense_counterfactual`: the model answers a counterfactual question; chrF.
* `farsense_explanation`: the model answers a "why" question; chrF.

Options and labels differ in length, so for the two log-likelihood tasks `acc_norm`, which normalizes
the log-likelihood by length, is the recommended accuracy.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The paper prompts models for JSON answers and reports exact-match accuracy for the classification
tasks and the mean of BLEU-4, ROUGE-L, BERTScore and embedding similarity for the free-form tasks
(scripts in the GitHub repository). These tasks use log-likelihood scoring and chrF instead, so the
numbers are not directly comparable to the paper's tables.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
