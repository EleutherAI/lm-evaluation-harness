# LiveCodeBench

### Paper

**LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code**

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica

UC Berkeley, MIT, Cornell University

https://arxiv.org/abs/2403.07974

### Description

LiveCodeBench is a **continuously-updated code evaluation benchmark** that prevents test set contamination by dating all problems and collecting new ones monthly from programming contests (LeetCode, AtCoder, CodeForces).

Unlike static benchmarks like HumanEval, LiveCodeBench:
- **Updates monthly** — new problems released every month
- **Prevents contamination** — each problem has a release date, so you can evaluate models only on unseen problems
- **Holistic evaluation** — measures code generation, self-repair, test output prediction, and code execution
- **High-quality problems** — curated from professional programming contests

### Current Task: Code Generation

This task evaluates **code generation** — can the model write correct code to solve a programming problem?

**Dataset**: `livecodebench/code_generation_lite` on HuggingFace
- **1,055 problems** from May 2023 to April 2025
- **Problems from**: LeetCode, AtCoder, CodeForces
- **Lite version**: Pruned test cases for faster evaluation (maintains similar quality signal)

### Metrics

| Metric | Description |
|---|---|
| `pass@1` | Does the first generated code pass all test cases? |
| `pass@5` | Do any of the top 5 generated codes pass all test cases? |

`pass@k` is the standard metric for code generation benchmarks. It measures: **1 - (failures on all k attempts) / (total problems)**.

For a single problem: **pass@k = 1 if any of the k outputs is correct, 0 otherwise**.

### Evaluation Approach

**Code Execution**: The task runs generated code against test cases using Python's subprocess module with:
- **Timeout protection**: 10 seconds per test case (configurable)
- **Safe execution**: Isolated via temporary files
- **Flexible I/O handling**: JSON, lists, dicts, strings, numbers

**Output Matching**: Lenient comparison handles:
- Case-insensitive matching
- Whitespace normalization
- JSON structure comparison
- Floating-point tolerance (1e-4)

### Preventing Contamination

LiveCodeBench problems are annotated with **release dates**. To evaluate on unseen problems:

```bash
# Evaluate on problems released after a model's training cutoff
# (Filter the dataset by release_date > training_date)

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks livecodebench_codegeneration \
    --filter "release_date > 2024-01-01"
```

This ensures models are tested on truly new problems, preventing inflated scores from training data leakage.

### Grading Details

**Test Case Format**:
```json
{
  "input": "[1, 2, 3]",  // JSON-serialized input(s)
  "output": "6"           // Expected output
}
```

**Execution Flow**:
1. Parse generated code
2. For each test case:
   - Run the code with the test input
   - Capture output
   - Compare with expected output (lenient matching)
3. Code passes if **all test cases match**
4. Metric: Did any of the k attempts pass?

### Example Usage

```bash
# Zero-shot (matches original paper setup)
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks livecodebench_codegeneration \
    --device cuda:0 \
    --batch_size 8

# Few-shot (exploratory)
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks livecodebench_codegeneration \
    --num_fewshot 5 \
    --device cuda:0

# With release date filtering
lm_eval --model hf \
    --model_args pretrained=GPT-4 \
    --tasks livecodebench_codegeneration \
    --filter "release_date >= 2024-09-01"
```

### Dataset Versions

LiveCodeBench provides timestamped versions:
- `release_v1`: May 2023 - Mar 2024 (400 problems)
- `release_v2`: May 2023 - May 2024 (511 problems)
- `release_v3`: May 2023 - Jul 2024 (612 problems)
- `release_v4`: May 2023 - Sep 2024 (713 problems)
- `release_v5`: May 2023 - Jan 2025 (880 problems)
- `release_v6`: May 2023 - Apr 2025 (1055 problems)

Users can specify versions via `--release_version` in the lcb_runner tool, or filter the dataset by release date in lm-eval-harness.

### Citation

```bibtex
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  journal   = {arXiv preprint arXiv:2403.07974},
  year      = {2024}
}
```

### Related Work

For broader code evaluation, also see:
- [EvalPlus](https://github.com/opencompass/EvalPlus) — enhanced test cases
- [BigCode Evaluator](https://github.com/bigcode-project/bigcode-evaluation-harness) — holistic code evaluation
- [APPS](https://github.com/hendrycks/apps) — application-driven benchmark

### Leaderboard

See the official LiveCodeBench leaderboard: https://livecodebench.github.io/leaderboard.html
