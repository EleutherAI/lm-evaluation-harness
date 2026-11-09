# MMLU Early Training

This is a update of the MMLU benchmark (Hendrycks et al.) to evaluate models at early training stages. MMLU consists of multiple-choice questions from various branches of knowledge, such as humanities, social sciences, hard sciences.  The dataset was created by selecting the 50% "easiest questions", that is, question in which there is a learning signal at the early training stages. Furthermore, MMLU is an multiple-choice dataset, and in this task, choices are not present in the prompt, and the target is the complete choice text, not only the choice letter. To further ensure signal at early training stages, the metric is defined as the difference of the log-likelihood of the correct choice and the average log-likelihood among incorrect choices.

### Groups, Tags, and Tasks

#### Groups
- Not part of a group yet.
#### Tasks
- `mmlu_early_training`

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
