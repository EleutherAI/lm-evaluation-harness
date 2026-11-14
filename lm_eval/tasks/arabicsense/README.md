# ArabicSense

## Paper

Title: `ArabicSense: A Benchmark for Evaluating Commonsense Reasoning in Arabic with Large Language Models`

Paper: https://aclanthology.org/2025.wacl-1.1/

ArabicSense tests commonsense reasoning in Modern Standard Arabic with three tasks built from
Arabic Wikipedia: telling which of two statements goes against common sense (Task A), picking the
reason why a statement is nonsensical among three candidates (Task B), and writing that reason
(Task C). The test set has 848 items per task.

Homepage: https://github.com/KamyarZeinalipour/ArabicSense

Data: [Task A](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense),
[Task B](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense-task2),
[Task C](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense-task3)

### Citation

```text
@inproceedings{lamsiyah-etal-2025-arabicsense,
    title = "{A}rabic{S}ense: A Benchmark for Evaluating Commonsense Reasoning in {A}rabic with Large Language Models",
    author = "Lamsiyah, Salima and Zeinalipour, Kamyar and El amrany, Samir and Brust, Matthias and
      Maggini, Marco and Bouvry, Pascal and Schommer, Christoph",
    booktitle = "Proceedings of the 4th Workshop on {A}rabic Corpus Linguistics ({WACL}-4)",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.wacl-1.1/",
    pages = "1--11"
}
```

### Groups, Tags, and Tasks

#### Groups

* `arabicsense`: Tasks A and B, accuracy averaged over items.

#### Tasks

* `arabicsense_validation`: Task A. Two statements; the model picks the one that goes against common sense (accuracy, F1).
* `arabicsense_explanation`: Task B. A nonsensical statement and three candidate reasons; the model's log-likelihood of each reason picks the answer (accuracy; `acc_norm`, normalized by length, is recommended because the reasons differ in length).
* `arabicsense_generation`: Task C. The model writes the reason; scored by sentence chrF against the closest of the three reference reasons.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

The paper evaluates fine-tuned encoders and prompted LLMs with F1 (Tasks A and B) and BERTScore
(Task C). These tasks use log-likelihood scoring for A and B and chrF for C, so the numbers are not
directly comparable to the paper's tables.

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial version.
