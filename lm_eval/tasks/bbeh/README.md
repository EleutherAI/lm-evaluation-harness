# BBEH (BIG-Bench Extra Hard)

### Paper

Title: `BIG-Bench Extra Hard`

BIG-Bench Extra Hard (BBEH) replaces each of the 23 tasks in BIG-Bench Hard
(BBH) with a novel, substantially harder variant, targeting the limits of
current LLM reasoning.

Homepage: https://github.com/google-deepmind/bbeh

### Citation

```
@article{kazemi2025bigbenchextrahard,
  title={BIG-Bench Extra Hard},
  author={Kazemi, Mehran and others},
  journal={arXiv preprint arXiv:2502.19187},
  year={2025}
}
```

### Groups and Tasks

#### Groups

- `bbeh`: all 23 BBEH subtasks (full set, 4520 examples), aggregated with
  size-weighted mean `exact_match` (i.e. micro average).

#### Tasks

- `bbeh_{subtask}` for each of the 23 subtasks (e.g. `bbeh_boardgame_qa`,
  `bbeh_word_sorting`, `bbeh_zebra_puzzles`, ...).
- `bbeh_mini`: the official 460-example BBEH-mini subset as a single task
  (micro average), directly comparable to the "BBEH Mini" column of the
  [official leaderboard](https://github.com/google-deepmind/bbeh/blob/main/leaderboard.md).

Data source: the official [`BBEH/bbeh`](https://huggingface.co/datasets/BBEH/bbeh)
dataset (one flat split; each subtask is selected from the `task` column via
`process_docs`, and `bbeh_mini` is selected via the `mini` flag).

### Implementation notes

- **Chain-of-thought, `generate_until`.** The prompt asks the model to reason and
  end with `The answer is: <answer>`. BBEH answers are heterogeneous across
  subtasks (multiple-choice `(A)`-style, numbers, words, lists), so generation is
  not truncated early.
- **Scoring mirrors the official BBEH evaluator** (`utils.py`, adapted from
  [`google-deepmind/bbeh/.../evaluate.py`](https://github.com/google-deepmind/bbeh/blob/main/bbeh/evaluate.py)):
  extract the answer after the `The answer is` prefix, strip LaTeX, then
  `fuzzy_match` (handles `(a)`/`a`, numeric equality, quotes, brackets) via a
  `process_results` hook.
- Best run with an instruct model and `--apply_chat_template`.

### Reproducibility / validation

To validate against the reference, run `bbeh_mini` on a model listed on the
official leaderboard and compare the `exact_match` to the "BBEH Mini" column
(e.g. Llama-3.1-8B-Instruct ≈ 11.5).

### Checklist

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] Does the original paper provide a reference implementation? Scoring in
    `utils.py` mirrors the official `evaluate.py`; `bbeh_mini` reproduces the
    leaderboard's mini setup for checking against published numbers.
