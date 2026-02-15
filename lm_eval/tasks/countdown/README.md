# Countdown Task

## Overview

The Countdown task evaluates a model's ability to solve arithmetic puzzles. Given a list of integers and a target number, the model must generate an arithmetic equation using each number exactly once with basic operations (+, -, *, /) such that the result equals the target number.

## Dataset

**Homepage**: https://huggingface.co/datasets/Stephen-Xie/Countdown

**Dataset Statistics**:
- Total train examples: 449,470
- Total test examples: 1000
- Numbers per problem: 3-4 integers
- Operations allowed: +, -, *, /, and parentheses

**Dataset Fields**:
- `nums`: List of integers to use
- `target`: Target value to reach

### Citation

```latex
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```

### Groups, Tags, and Tasks

#### Groups

* `countdown`: Arithmetic reasoning task requiring unique use of numbers

#### Tags

* `arithmetic`: Tasks involving arithmetic operations
* `math`: Mathematical reasoning tasks

#### Tasks

* `countdown`: Generate an arithmetic equation using given numbers exactly once to reach a target value

## Example

**Input Prompt**:
```
Given the numbers 41, 70, 18, 35, how can you use each number exactly once 
with basic arithmetic operations (+, -, *, /) to reach the target 46?

Answer:
```

**Valid Solutions**:
- `(70 - 41) + (35 - 18)` = 46
- `70 + 35 - 41 - 18` = 46
- `(70 + 35) - (41 + 18)` = 46

## Scoring

The task uses a custom scoring function (`utils.compute_score`):

| Score | Criteria | Example |
|-------|----------|---------|
| **1.0** | Correct equation: uses all numbers exactly once AND evaluates to target | `(70 - 41) + (35 - 18)` for target 46 |
| **0.0** | No valid equation found | `I don't know` |

## Usage

To evaluate a model on the Countdown task:

```bash
lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen3-8B,tensor_parallel_size=4,max_length=32768,max_gen_toks=8192,reasoning_parser=qwen3,think_end_token="</think>" \
    --tasks countdown \
    --device cuda \
    --batch_size auto \
    --apply_chat_template
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Dataset available at HuggingFace: Jiayi-Pan/Countdown-Tasks-3to4-Unique
  * [x] Implementation tested and validated

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?
