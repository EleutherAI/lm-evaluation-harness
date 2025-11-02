# Countdown Task

## Overview

The Countdown task evaluates a model's ability to solve arithmetic puzzles. Given a list of integers (typically 3-4 numbers) and a target number, the model must generate an arithmetic equation using each number exactly once with basic operations (+, -, *, /) such that the result equals the target number.

This task tests:
- **Arithmetic reasoning**: Understanding of basic mathematical operations
- **Constraint satisfaction**: Using each number exactly once
- **Problem-solving**: Finding valid combinations to reach the target

## Dataset

**Homepage**: https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4-Unique

**Dataset Statistics**:
- Total examples: 449,570
- Split: train (used for evaluation)
- Numbers per problem: 3-4 integers
- Operations allowed: +, -, *, /

**Dataset Fields**:
- `nums`: List of integers to use
- `target`: Target value to reach

### Citation

```text
If available, BibTeX-formatted citation goes here
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

The task uses a custom scoring function (`utils.compute_score`) with three-tier scoring:

| Score | Criteria | Example |
|-------|----------|---------|
| **1.0** | Correct equation: uses all numbers exactly once AND evaluates to target | `(70 - 41) + (35 - 18)` for target 46 |
| **0.1** | Format credit: valid equation format but doesn't meet all requirements | `70 + 41` (not all numbers used) |
| **0.0** | No valid equation found | `I don't know` |

### Scoring Process

1. **Extract equation** from model output (supports multiple formats):
   - Direct: `(70 - 41) + (35 - 18)`
   - With tags: `<answer>(70 - 41) + (35 - 18)</answer>`
   - With result: `(70 - 41) + (35 - 18) = 46`
   - With prefix: `Answer: (70 - 41) + (35 - 18)`

2. **Validate** that equation uses each available number exactly once

3. **Evaluate** the equation safely (with security checks)

4. **Compare** result to target (with floating-point tolerance)

## Usage

To evaluate a model on the Countdown task:

```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks countdown \
    --batch_size 8 \
    --num_fewshot 0 \
    --limit 100
```

### Parameters

- `--num_fewshot`: Number of few-shot examples (default: 0)
- `--limit`: Number of examples to evaluate (useful for quick tests)
- `--batch_size`: Batch size for evaluation

## Testing

The implementation includes comprehensive tests. To run them:

```bash
cd lm_eval/tasks/countdown
python test_countdown.py  # Run full test suite
python demo.py            # Run interactive demo
```

### Test Coverage

- ✓ Prompt generation (`doc_to_text`)
- ✓ Equation extraction from various formats
- ✓ Number validation (each used exactly once)
- ✓ Safe equation evaluation
- ✓ Scoring with edge cases
- ✓ Result processing

## Implementation Notes

### Security
- Uses restricted `eval()` with safety checks
- Only allows numbers, operators (+, -, *, /), parentheses, and whitespace
- Protects against division by zero
- Validates results (checks for inf/nan)

### Robustness
- Handles multiple output formats (with/without tags, prefixes, results)
- Field name flexibility (supports both 'nums'/'numbers' and 'target'/'target_number')
- Comprehensive error handling
- Floating-point tolerance for comparisons

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Dataset available at HuggingFace: Jiayi-Pan/Countdown-Tasks-3to4-Unique
  * [x] Implementation tested and validated

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

## Changelog

* **Version 1.0** (Initial implementation):
  - Added support for Countdown-Tasks-3to4-Unique dataset
  - Implemented three-tier scoring system
  - Added comprehensive equation extraction and validation
  - Included safety checks for equation evaluation
  - Created test suite and demo script
