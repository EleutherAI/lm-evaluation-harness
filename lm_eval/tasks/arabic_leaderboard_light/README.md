# Arabic Leaderboard Light

Title: Open Arabic LLM Leaderboard Light

This leaderboard follows all the details as in [`arabic_leaderboard_complete`](../arabic_leaderboard_complete), except that a light version - 10% random sample of the test set of each benchmark - is used to test the language models.

NOTE: In ACVA benchmark, there is Yemen subset, and it is a small dataset - it has only 10 samples in the test split. So, for this specific subset dataset, to have more reliable results, we consider the original dataset, instead of 10% of its test samples.

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
