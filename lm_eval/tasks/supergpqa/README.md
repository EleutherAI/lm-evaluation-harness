# SuperGPQA

This port uses the official zero-shot and fixed five-shot prompts and answer
extraction from
[`SuperGPQA/SuperGPQA@e31e8bf`](https://github.com/SuperGPQA/SuperGPQA/tree/e31e8bf3e7089aef808be5ddecafd4538dee6e41).
It loads `m-a-p/SuperGPQA` at immutable dataset revision
`4430d4458112c7d4497fdcf94d7cc223313d6acf`.

Selectors:

- `supergpqa`: official zero-shot prompt;
- `supergpqa_five_shot`: official fixed five-shot prompt.

Both selectors report sample accuracy, macro accuracy over subfields, fields,
and disciplines, difficulty-sliced sample accuracy, miss rate, and extraction
error rate.

SuperGPQA is a composite database made available under ODC-BY. It contains
material derived from other benchmarks with their own licenses. Follow the
attribution and underlying-content notices in the pinned upstream README.
