# theoremqa

### Paper

Title: `TheoremQA: A Theorem-driven Question Answering Dataset`

Abstract: `The recent LLMs like GPT-4 and PaLM-2 have made tremendous progress in solving fundamental math problems like GSM8K by achieving over 90% accuracy. However, their capabilities to solve more challenging math problems which require domain-specific knowledge (i.e. theorem) have yet to be investigated. In this paper, we introduce TheoremQA, the first theorem-driven question-answering dataset designed to evaluate AI models' capabilities to apply theorems to solve challenging science problems.`

Homepage: https://github.com/TIGER-AI-Lab/TheoremQA

### Citation

```
@inproceedings{chen-etal-2023-theoremqa,
    title     = {{T}heorem{QA}: A Theorem-driven Question Answering Dataset},
    author    = {Wenhu Chen and Ming Yin and Max Ku and Pan Lu and Yixin Wan and Xueguang Ma and Jianyu Xu and Xinyi Wang and Tony Xia},
    booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    year      = {2023},
    pages     = {7889--7901},
}
```

### Groups, Tags, and Tasks

#### Tasks

* `theoremqa`: 5-shot chain-of-thought evaluation over the text-only questions, scored with the reference repository's type-aware answer matching.

### Implementation notes

**Text-only subset.** The dataset ships 800 test rows, 53 of which carry a
`Picture`. A text-only model cannot answer those, so `process_docs` drops the
column and keeps the remaining **747** rows. The column is removed rather than
ignored because `datasets` decodes `Image` features on access and raises
`ImportError` without Pillow, which would otherwise break loading for anyone
who installed the harness without the `dev` extra. Reported numbers are
therefore not comparable to multimodal runs over the full 800.

**Few-shot prompts are fixed, not sampled.** The dataset has only a `test`
split, so sampling few-shot context would leak evaluation rows. The five
prompts in `utils.list_fewshot_samples` are carried verbatim from the reference
repository's `examples.py`, and none of the five questions appears in the 800
test rows. One repair was needed: that file uses a non-raw string, so the third
example's `\frac` reached runtime as a form feed.

**Scoring** follows the reference `utils.py` / `number_utils.py`: comparison is
driven by each row's `Answer_type`, floats match inside a 4% relative window,
integers match after rounding, and lists match elementwise after sorting.
Two deliberate deviations: the reference resolves predictions with
`latex2sympy` and `eval()`, while this task parses numbers directly, so it adds
no dependency and never executes model output. A prediction whose only form is
LaTeX needing a CAS is therefore scored incorrect rather than evaluated.

### Checklist

For adding novel benchmarks/datasets to the library:
* [X] Is the task an existing benchmark in the literature?
  * [X] Have you referenced the original paper that introduced the task?
  * [X] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
