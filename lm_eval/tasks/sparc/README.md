# SPaRC

## Paper

Title: `SPaRC: A Spatial Pathfinding Reasoning Challenge` (EMNLP 2025)

Abstract: https://aclanthology.org/2025.emnlp-main.526/ (arXiv: https://arxiv.org/abs/2505.16686)

SPaRC has 1,000 2D grid pathfinding puzzles based on the game *The Witness*. The model must draw a line
from a start node to an end node that obeys every rule on the grid (dots, gaps, stones, stars, triangles,
polyominoes). The test split has 500 puzzles at difficulty levels 1 to 5. In the paper, humans solve 98.0%
(94.5% of the hardest level) and the best model, o4-mini, solves 15.8% (1.1% of the hardest level).

Homepage: https://sparc.gipplab.org/ · Code: https://github.com/lkaesberg/SPaRC ·
Dataset: https://huggingface.co/datasets/lkaesberg/SPaRC

### Citation

```text
@inproceedings{kaesberg-etal-2025-sparc,
  title     = "{SP}a{RC}: A Spatial Pathfinding Reasoning Challenge",
  author    = "Kaesberg, Lars Benedikt and Wahle, Jan Philip and Ruas, Terry and Gipp, Bela",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  month     = nov,
  year      = "2025",
  address   = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url       = "https://aclanthology.org/2025.emnlp-main.526/",
  doi       = "10.18653/v1/2025.emnlp-main.526",
  pages     = "10359--10390"
}
```

### Groups, Tags, and Tasks

#### Groups

* None.

#### Tasks

* `sparc`: The full 500-puzzle test split (main variant). It reports overall `acc` and `invalid_path`, plus
  `acc_level_1` … `acc_level_5` for each difficulty level (86 / 118 / 121 / 86 / 89 puzzles). Puzzles are
  ordered round-robin by difficulty level, so `--limit n` covers all levels evenly.
* `sparc_hard`: Only the 89 level-5 puzzles, which the paper calls "hard".

#### Metrics

* `acc` / `acc_level_<k>`: The fraction of puzzles where the extracted path exactly matches one of the puzzle's stored
  solutions. The dataset stores every valid solution, enumerated by brute force.
* `invalid_path` (lower is better): The fraction of answers that are not a valid line from start to end. An
  answer counts as invalid if it has the wrong start or end, is disconnected, crosses itself, or passes
  through a rule cell. This is the "Invalid Path" column of Table 7 in the paper.

#### Prompt and scoring

The prompt is the paper's single-shot prompt. It contains one worked example inside the prompt, so
`num_fewshot` is 0. The system message is the task `description`. The model reasons step by step, then
writes `####` followed by the coordinate list. The answer is extracted from the text after the last `####`.
`utils.py` holds verbatim copies of the reference prompt builder
([`sparc/prompts/single_shot.py`](https://github.com/lkaesberg/SPaRC/blob/9f2a3d6/sparc/prompts/single_shot.py))
and validator ([`sparc/validation.py`](https://github.com/lkaesberg/SPaRC/blob/9f2a3d6/sparc/validation.py)).
The copies were checked to produce identical prompts and scores on all 500 test puzzles.

Generation is greedy (`temperature: 0.0`) with `max_gen_toks: 32768`, because reasoning models often
produce more than 10k tokens per puzzle. The paper does not state a temperature, and the reference CLI
defaults to `--temperature 1.0`. Override with `--gen_kwargs` if needed. For chat models, run with
`--apply_chat_template`. For reasoning models that put `<think>` blocks in the content, pass
`think_end_token=</think>` in `--model_args`.

Example:

```bash
lm_eval --model local-chat-completions \
  --model_args model=<model>,base_url=<server>/v1/chat/completions,num_concurrent=8 \
  --tasks sparc --apply_chat_template
```

### Results

Full test split (500 puzzles), run on 2026-10-13 with
`--model local-chat-completions` (OpenRouter, `google/gemma-3-27b-it`), `--apply_chat_template` and
`--gen_kwargs max_gen_toks=8192`. The paper numbers are from Tables 6 and 7 of the SPaRC paper.

| Metric | Paper | This task |
|---|---|---|
| `acc` | 1.2% | 0.8% (4/500) |
| `acc_level_1` … `acc_level_5` | 3.5 / 0.8 / 0.8 / 0.0 / 1.1% | 1.2 / 0.8 / 0.8 / 1.2 / 0.0% |
| `invalid_path` | 88.0% | 84.2% |

The overall accuracy is within 0.4 points of the paper. The paper used a different serving setup and does
not state its sampling settings.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    The reference implementation is the `sparc` CLI in https://github.com/lkaesberg/SPaRC. `utils.py` copies its
    prompt and validator verbatim, and the copies were checked for identical outputs on all 500 test puzzles.

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
  `sparc` is the main variant; `sparc_hard` is its level-5 subset.
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
  `sparc` matches the paper's main single-shot text setting (Table 6, including the per-level columns).
  `sparc_hard` matches the paper's "hard" numbers.

### Changelog

* [Oct 13, 2026] Version 1.0: Initial release.
