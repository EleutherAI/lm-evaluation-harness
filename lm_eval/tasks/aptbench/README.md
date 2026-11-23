# APTBench

## Paper

Title: `APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training`

Paper: https://arxiv.org/abs/2510.24397

APTBench evaluates the agentic potential of base LLMs during pre-training by converting real-world agent tasks and execution trajectories into lightweight multiple-choice and text-completion tasks across software engineering, deep research, tool use, and agentic math.

Homepage: https://github.com/TencentYoutuResearch/APTBench

### Citation

```bibtex
@misc{qin2025aptbench,
      title={APTBench: Benchmarking Agentic Potential of Base LLMs During Pre-Training},
      author={Jiarui Qin and Yunjia Xi and Junjie Huang and Renting Rui and Di Yin and Weiwen Liu and Yong Yu and Weinan Zhang and Xing Sun},
      year={2025},
      eprint={2510.24397},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.24397},
}
```

## Usage & Configuration

### Data & Prompt Loading

Because APTBench does not currently host an official dataset on the Hugging Face Hub, tasks default to downloading the raw JSONL datasets and prompt templates from the pinned upstream commit (`9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc`) on GitHub (`TencentYoutuResearch/APTBench`), cached through `datasets.DownloadManager`.

Quick-start example with `Qwen/Qwen3-1.7B-Base` (using its native `32,768` context window):

```bash
# Quick-start with a local clone of TencentYoutuResearch/APTBench
lm-eval --model vllm \
  --model_args pretrained=Qwen/Qwen3-1.7B-Base,truncation_side=middle \
  --tasks aptbench \
  --metadata '{"aptbench_dir": "/path/to/APTBench"}'
```

Long-context 128K evaluation example using a model with native `max_position_embeddings=131072` (e.g., `Qwen/Qwen2.5-7B`):

```bash
lm-eval --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-7B,max_model_len=131072,truncation_side=middle \
  --tasks aptbench
```

Supported namespaced override options (`--metadata` keys / environment variables):
* `aptbench_dir` / `APTBENCH_DIR`: Root directory of a local `APTBench` clone (reads `data/` and `code/prompts/`).
* `aptbench_data_dir` / `APTBENCH_DATA_DIR`: Directory containing `<dataset>/<task>/input_data.jsonl`.
* `aptbench_prompt_dir` / `APTBENCH_PROMPT_DIR`: Directory containing the `.txt` prompt templates.
* `aptbench_revision` / `APTBENCH_REVISION`: GitHub commit SHA or ref to download from (defaults to `9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc`).

> [!NOTE]
> All source override keys use the `aptbench_*` prefix so standard model arguments passed via `--model_args` (such as `revision=step10000` for pre-training checkpoint evaluation) never collide with APTBench dataset or prompt resolution.

### Long-Context Truncation & Backend Behavior

* **Architectural boundary**: `doc_to_text()` always emits the full pre-truncation prompt; token budgeting and truncation are delegated to the model backend.
* **1:1 parity scope**: Pre-truncation prompt formatting, extraction regexes, and `max_gen_toks` match `APTBench/code/predict.py` 1:1. All tasks use greedy decoding (`do_sample: false`, `temperature: 0.0`, `until: []`) and `num_fewshot: 0` because the 0-, 2-, or 3-shot exemplars from `APTBench/code/prompts/` are embedded verbatim in each prompt template.
* **Backend support**: `vllm` and `trtllm` support `truncation_side=middle` (preserving both task instructions/few-shot examples at the front and the target question at the end). `hf` (`HFLM`), `sglang`, and `local-completions` do **not** support `truncation_side=middle` (`hf` truncates from the left, which can drop instructions and few-shot exemplars on long prompts).
* **Context-length impact** (measured with upstream `assets/llama3_tokenizer` across all 12,725 items; prompts reach up to ~992K chars / ~235K tokens):
  * **128K (`max_model_len=131072`)**: Only 8 items across 3 tasks exceed 120K/131K tokens (`env_setup_plan`: 3/437 = 0.7%, `deepresearch_openend_citation_en`: 1/173 = 0.6%, `deepresearch_openend_citation_zh`: 4/189 = 2.1%).
  * **`<=32K` (`32,768` tokens, e.g., `Qwen/Qwen3-1.7B-Base` without YaRN)**: 1,176 items across 9 tasks exceed 32K tokens (100% of `deepresearch_openend_citation_zh`, `issue_fix_plan`, and `issue_fix_action`), making backend choice and truncation behavior significant.
* Always record the model backend, `max_model_len`, and truncation setting when reporting results.

### Request Caching Caveat

`lm-eval`'s `--cache_requests true` request cache key does not include `aptbench_*` override metadata. When switching between local, custom, or remote APTBench sources with request caching enabled, pass `--cache_requests refresh` or isolate `LM_HARNESS_CACHE_PATH` per source.

### Metrics, Paper Scope & Reference Comparison

* **Paper vs. repository scope**: The APTBench paper evaluates **SWE** (8 tasks in Table 2) and **DeepResearch** (9 tasks in Table 3). The **Tool** (4 tasks) and **Agentic Math** (3 tasks) suites were added in the pinned upstream GitHub repository (`9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc`).
* **`openend_quality_en` sample count**: The pinned upstream release contains **110** items (`2,254` total DeepResearch items), whereas Paper Table 5 lists **111** items (`2,255` total). This integration evaluates all 110 items present in the pinned release.
* **Reported metrics**: Every task reports a unified `score` metric (for unweighted macro-averaging via `weight_by_size: false`) alongside its native metric (`acc` for multiple-choice / true-false / multi-label citation tasks; `exact_match` for text-completion tasks).
  * `aptbench_deepresearch_summ_ans_en` and `aptbench_deepresearch_summ_ans_zh` also report `exact_match_flex` (stripping surrounding single/double quotes from the extracted prediction, matching `code/results/deepresearch/get_results.py`). When comparing against Paper Table 3 `EM`, note which of `exact_match` or `exact_match_flex` is used.
  * For `tool/*_api_select`, this port reports `exact_match` (alongside `score`) rather than the internal `ACC` label in `code/results/tool/get_results.py`, keeping naming consistent across text-completion tasks.
* **Group average formulas (Table 2 vs. Table 3)**:
  * `aptbench_swe` `score` (8-task unweighted macro-average of ACC/EM) matches the `AVG` column in Paper Table 2 identically.
  * `aptbench_deepresearch` `score` averages **9** ACC/EM metrics, whereas Paper Table 3's `Avg` averages **11** columns (7 ACC + 2 EM + 2 ROUGE-1 F1 scores on `summ_ans_en` and `summ_ans_zh`). **These two averages use different formulas and are not directly comparable** (differences vary in sign and magnitude by model, e.g., `28.52%` 11-col vs. `25.05%` 9-col for `Qwen3-1.7B`; `66.42%` 11-col vs. `66.74%` 9-col for `DeepSeek-V3.1`). Individual per-task ACC/EM metrics remain directly comparable to Table 3 columns. To reproduce the full 11-column Table 3 `Avg`, post-process `--log_samples` outputs (`samples_*.jsonl`) using upstream `assets/llama3_tokenizer` and the `rouge` package for `summ_ans_en` and `summ_ans_zh` ROUGE-1 F1.
* **Empty model outputs**: `lm-eval` scores empty generations (`""`) as `0.0` in the full dataset denominator, whereas `APTBench/code/predict.py` skips writing empty outputs. When comparing against upstream runs, verify completed UUIDs / skipped empty responses.
* **Running the reference-parity test**:

```bash
APTBENCH_DIR=/path/to/APTBench pytest -vv tests/test_aptbench.py
```

## Groups, Tags, and Tasks

### Groups

* `aptbench`: Harness-added unweighted macro-average across all 24 leaf tasks (`weight_by_size: false`; `lm-eval` flattens nested groups, so domain weights are proportional to task count: SWE 8/24, DeepResearch 9/24, Tool 4/24, Agentic Math 3/24).
  * `aptbench_swe` (alias `swe`): Unweighted macro-average across all 8 SWE leaf tasks (3 `env_setup` + 5 `issue_fix`), matching the `AVG` column in Table 2 of the paper.
    * `aptbench_env_setup` (alias `env_setup`): Environment setup tasks (3 tasks).
    * `aptbench_issue_fix` (alias `issue_fix`): Issue fixing tasks (5 tasks).
  * `aptbench_deepresearch` (alias `deepresearch`): Unweighted macro-average across all 9 DeepResearch ACC/EM tasks (see formula note above vs. Paper Table 3's 11-column `Avg`).
  * `aptbench_tool` (alias `tool`): Unweighted macro-average across all 4 Tool tasks.
  * `aptbench_agentic_math` (alias `agentic_math`): Unweighted macro-average across all 3 Agentic Math tasks.

### Tags

* `aptbench_tasks`: All 24 APTBench tasks.
* `aptbench_swe_tasks`: All 8 SWE tasks (`env_setup` + `issue_fix`).
* `aptbench_env_setup_tasks`: All 3 `env_setup` tasks.
* `aptbench_issue_fix_tasks`: All 5 `issue_fix` tasks.
* `aptbench_deepresearch_tasks`: All 9 `deepresearch` tasks.
* `aptbench_tool_tasks`: All 4 `tool` tasks.
* `aptbench_agentic_math_tasks`: All 3 `agentic_math` tasks.

### Tasks

| Task Name | Alias | Origin | Samples | Embedded Shots | `max_gen_toks` | Metrics | Description |
|-----------|-------|--------|---------|----------------|----------------|---------|-------------|
| `aptbench_env_setup_plan` | `env_setup_plan` | Paper Table 2 | 437 | 3-shot | 10 | `score`, `acc` | Selects the correct repository environment-setup plan from repository context. |
| `aptbench_env_setup_action` | `env_setup_action` | Paper Table 2 | 1084 | 3-shot | 30 | `score`, `exact_match` | Predicts the next bash setup command given the plan and previously executed commands. |
| `aptbench_env_setup_error` | `env_setup_error` | Paper Table 2 | 147 | 3-shot | 10 | `score`, `acc` | Identifies the execution plan that resolves a reported environment-setup error. |
| `aptbench_issue_fix_locate` | `issue_fix_locate` | Paper Table 2 | 283 | 3-shot | 10 | `score`, `acc` | Identifies the buggy code snippet described in a GitHub issue statement. |
| `aptbench_issue_fix_fix_patch` | `issue_fix_fix_patch` | Paper Table 2 | 232 | 3-shot | 10 | `score`, `acc` | Selects the patch that fixes the issue described in a GitHub issue statement. |
| `aptbench_issue_fix_plan` | `issue_fix_plan` | Paper Table 2 | 243 | 3-shot | 10 | `score`, `acc` | Chooses the next step plan given an issue-fixing agent trajectory. |
| `aptbench_issue_fix_action` | `issue_fix_action` | Paper Table 2 | 241 | 3-shot | 10 | `score`, `exact_match` | Generates the next tool command from an issue-fixing trajectory. |
| `aptbench_issue_fix_test_patch` | `issue_fix_test_patch` | Paper Table 2 | 1060 | 3-shot | 10 | `score`, `acc` | Selects the test patch that reproduces the reported problem statement. |
| `aptbench_deepresearch_plan_en` | `deepresearch_plan_en` | Paper Table 3 | 614 | 3-shot | 10 | `score`, `acc` | Selects the correct next search/browse action in English closed-ended research trajectories. |
| `aptbench_deepresearch_plan_zh` | `deepresearch_plan_zh` | Paper Table 3 | 417 | 3-shot | 10 | `score`, `acc` | Selects the correct next search/browse action in Chinese closed-ended research trajectories. |
| `aptbench_deepresearch_summ_ans_en` | `deepresearch_summ_ans_en` | Paper Table 3 | 212 | 3-shot | 30 | `score`, `exact_match`, `exact_match_flex` | Summarizes the final answer from an English search and browsing trajectory. |
| `aptbench_deepresearch_summ_ans_zh` | `deepresearch_summ_ans_zh` | Paper Table 3 | 138 | 3-shot | 30 | `score`, `exact_match`, `exact_match_flex` | Summarizes the final answer from a Chinese search and browsing trajectory. |
| `aptbench_deepresearch_openend_plan_en` | `deepresearch_openend_plan_en` | Paper Table 3 | 298 | 3-shot | 10 | `score`, `acc` | Selects the best report outline for an English open-ended research question. |
| `aptbench_deepresearch_openend_citation_en` | `deepresearch_openend_citation_en` | Paper Table 3 | 173 | 3-shot | 10 | `score`, `acc` | Identifies all article statements supported by the cited English webpage. |
| `aptbench_deepresearch_openend_citation_zh` | `deepresearch_openend_citation_zh` | Paper Table 3 | 189 | 3-shot | 10 | `score`, `acc` | Identifies all article statements supported by the cited Chinese webpage. |
| `aptbench_deepresearch_openend_quality_en` | `deepresearch_openend_quality_en` | Paper Table 3 | 110 | 2-shot | 10 | `score`, `acc` | Chooses the highest-quality English research report among four candidates. |
| `aptbench_deepresearch_openend_quality_zh` | `deepresearch_openend_quality_zh` | Paper Table 3 | 103 | 2-shot | 10 | `score`, `acc` | Chooses the highest-quality Chinese research report among four candidates. |
| `aptbench_tool_acebench_api_select` | `tool_acebench_api_select` | Repo | 697 | 3-shot | 30 | `score`, `exact_match` | Selects the target tool function name for an ACEBench user query. |
| `aptbench_tool_acebench_api_param` | `tool_acebench_api_param` | Repo | 1538 | 3-shot | 30 | `score`, `exact_match` | Generates the value for a named parameter of an ACEBench tool call. |
| `aptbench_tool_bfcl_v4_api_select` | `tool_bfcl_v4_api_select` | Repo | 197 | 3-shot | 30 | `score`, `exact_match` | Selects the target tool function name for a BFCL v4 user query. |
| `aptbench_tool_bfcl_v4_api_param` | `tool_bfcl_v4_api_param` | Repo | 565 | 3-shot | 30 | `score`, `exact_match` | Generates the value for a named parameter of a BFCL v4 tool call. |
| `aptbench_agentic_math_planning_single` | `agentic_math_planning_single` | Repo | 1448 | 3-shot | 10 | `score`, `acc` | Identifies the primary mathematical atomic capability required to solve a math problem. |
| `aptbench_agentic_math_feedback_tf` | `agentic_math_feedback_tf` | Repo | 1198 | 0-shot | 10 | `score`, `acc` | Judges whether a mathematical solution is `correct` or `wrong`. |
| `aptbench_agentic_math_action_cal` | `agentic_math_action_cal` | Repo | 1101 | 0-shot | 10 | `score`, `acc` | Selects the correct answer choice for an agentic math calculation question. |

## Checklist

For adding novel benchmarks/datasets to the library:

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

- [x] Is the "Main" variant of this task clearly denoted? (N/A - new task family)
- [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
- [x] Have you noted which, if any, published evaluation setups are matched by this variant?

## Changelog

* `v1.0`: Initial port of all 24 APTBench tasks (`env_setup`, `issue_fix`, `deepresearch`, `tool`, `agentic_math`) with ACC/EM evaluation matching `TencentYoutuResearch/APTBench` (`9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc`), implemented via `APTBenchTask` with namespaced `aptbench_*` override keys, strict required-field validation, and path-independent sample hashing.
