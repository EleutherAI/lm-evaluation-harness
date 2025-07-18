# Permutation Composition Benchmark

This suite of tasks evaluates a model's ability to perform permutation composition, a key measure of its state-tracking and symbolic reasoning capabilities. The theoretical basis for this benchmark is presented in the paper "The Illusion of State in State-Space Models."

The tasks use the `BeeGass/Group-Theory-Collection` dataset on the Hugging Face Hub.

## Overview

The benchmark consists of 94 permutation groups divided into two computational complexity classes:
- **TC⁰** (58 groups): Solvable groups that can be recognized by threshold circuits of constant depth
- **NC¹** (36 groups): Non-solvable groups requiring logarithmic depth circuits

Each task evaluates models on composing sequences of permutations of varying lengths (5 to 500 elements in increments of 5), providing 100 different performance metrics per group.

## Task Structure

### Main Task Groups
- `permutation_groups`: All 94 permutation groups
- `tc0`: All 58 TC⁰ (solvable) groups
- `nc1`: All 36 NC¹ (non-solvable) groups

### Individual Groups

#### TC⁰ Groups (Solvable)
- **Symmetric**: S3, S4
- **Alternating**: A3, A4
- **Cyclic**: C2-C30 (29 groups)
- **Dihedral**: D3-D20 (18 groups)
- **Quaternion**: Q8, Q16, Q32
- **Frobenius**: F20, F21
- **Klein four-group**: V4
- **Elementary abelian**: Z2_1-Z2_5, Z3_1-Z3_4, Z5_1-Z5_4
- **Projective Special Linear**: PSL2(2), PSL2(3)

#### NC¹ Groups (Non-Solvable)
- **Symmetric**: S5-S9
- **Alternating**: A5-A9
- **Projective Special Linear**: PSL2(4,5,7,8,9,11), PSL3(2,3,4,5)
- **Mathieu**: M11, M12

## Usage

```bash
# Evaluate on all groups
lm_eval --model hf --model_args pretrained=model_name --tasks permutation_groups

# Evaluate on TC⁰ groups only
lm_eval --model hf --model_args pretrained=model_name --tasks tc0

# Evaluate on NC¹ groups only
lm_eval --model hf --model_args pretrained=model_name --tasks nc1

# Evaluate on specific groups
lm_eval --model hf --model_args pretrained=model_name --tasks s3_composition,a5_composition
```

## Metrics

Each task reports 100 metrics corresponding to sequence lengths from 5 to 500 (in increments of 5). The metric names are the sequence lengths themselves (e.g., "5", "10", "15", ..., "500").

Metrics report accuracy (0-1) for samples at that sequence length, or -1 if no samples exist for that length.

## Expected Results

- Models not specifically trained on this task typically achieve 0% accuracy
- State-space models (SSMs) without explicit state-tracking mechanisms are expected to fail
- Chain-of-thought models may show better performance by maintaining intermediate states

### Task Validity Checklist

- [x] Is the task an existing benchmark in the literature?
  - [x] Have you referenced the original paper that introduced the task?
  - [x] The original paper provides the theoretical framework. The `BeeGass/Group-Theory-Collection` dataset is a new, concrete implementation of this concept. Its correctness can be verified independently via the permutation multiplication tables.