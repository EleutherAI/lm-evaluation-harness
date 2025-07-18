# Permutation Composition Benchmark

### Paper

Title: `The Illusion of State in State-Space Models`

Paper Link: https://arxiv.org/abs/2404.08819

Abstract: State-space models (SSMs) have emerged as a powerful model class for sequence modeling tasks. While recent works have shown that SSMs can solve simple in-context learning (ICL) tasks, their ability to perform more complex state-tracking tasks remains unclear. This work evaluates SSMs on permutation composition, a fundamental state-tracking task requiring models to compose sequences of permutations. 

Dataset: https://huggingface.co/datasets/BeeGass/Group-Theory-Collection

Dataset Generator: https://github.com/BeeGass/Group-Dataset-Generator

### Citation

```bibtex
@article{merrill2024illusion,
  title={The Illusion of State in State-Space Models},
  author={Merrill, William and Potts, Christopher and Kreider, Jackson},
  journal={arXiv preprint arXiv:2404.08819},
  year={2024}
}
```

### Overview

The Permutation Composition Benchmark evaluates a model's ability to perform permutation composition, a key measure of its state-tracking and symbolic reasoning capabilities. The benchmark consists of 94 permutation groups divided into two computational complexity classes based on their algebraic structure:

- **TC⁰** (72 groups): Solvable groups that can be recognized by threshold circuits of constant depth
- **NC¹** (22 groups): Non-solvable groups requiring logarithmic depth circuits

Each task evaluates models on composing sequences of permutations of varying lengths (5 to 500 elements in increments of 5), providing 100 different performance metrics per group. The composition follows the standard mathematical convention (right-to-left): p_n ∘ ... ∘ p_2 ∘ p_1.

### Groups and Tasks

#### Groups

* `permutation_groups`: All 94 permutation groups (includes both TC⁰ and NC¹)
* `tc0_groups`: All 72 TC⁰ (solvable) groups
* `nc1_groups`: All 22 NC¹ (non-solvable) groups

#### Individual Tasks

Each group has a corresponding task named `{group_name}_composition`. Tasks are categorized as follows:

**TC⁰ Groups (Solvable) - 72 groups:**
- **Symmetric**: `s3_composition`, `s4_composition`
- **Alternating**: `a3_composition`, `a4_composition`
- **Cyclic**: `c2_composition` through `c30_composition` (29 groups)
- **Dihedral**: `d3_composition` through `d20_composition` (18 groups)
- **Quaternion**: `q8_composition`, `q16_composition`, `q32_composition`
- **Frobenius**: `f20_composition`, `f21_composition`
- **Klein four-group**: `v4_composition`
- **Elementary abelian**: 
  - Z₂ᵏ: `z2_1_composition` through `z2_5_composition`
  - Z₃ᵏ: `z3_1_composition` through `z3_4_composition`
  - Z₅ᵏ: `z5_1_composition` through `z5_4_composition`
- **Projective Special Linear**: `psl2_2_composition`, `psl2_3_composition`

**NC¹ Groups (Non-Solvable) - 22 groups:**
- **Symmetric**: `s5_composition` through `s9_composition`
- **Alternating**: `a5_composition` through `a9_composition`
- **Projective Special Linear**: 
  - PSL(2,q): `psl2_4_composition`, `psl2_5_composition`, `psl2_7_composition`, `psl2_8_composition`, `psl2_9_composition`, `psl2_11_composition`
  - PSL(3,q): `psl3_2_composition`, `psl3_3_composition`, `psl3_4_composition`, `psl3_5_composition`
- **Mathieu**: `m11_composition`, `m12_composition`

### Usage

```bash
# Evaluate on all 94 groups
lm_eval --model hf --model_args pretrained=model_name --tasks permutation_groups

# Evaluate on TC⁰ groups only (72 groups)
lm_eval --model hf --model_args pretrained=model_name --tasks tc0_groups

# Evaluate on NC¹ groups only (22 groups)
lm_eval --model hf --model_args pretrained=model_name --tasks nc1_groups

# Evaluate on specific individual groups
lm_eval --model hf --model_args pretrained=model_name --tasks s3_composition,a5_composition,d10_composition

# Limit number of examples per task (useful for testing)
lm_eval --model hf --model_args pretrained=model_name --tasks permutation_groups --limit 100
```

### Task Format

Each task uses a loglikelihood evaluation format where the model must predict the correct permutation ID given a sequence of permutations to compose.

**Input Format:**
```
You are given a sequence of permutations from the group {group_name}, identified by their integer IDs. Your task is to compute their composed product.

The composition must be performed sequentially from right to left, following the standard mathematical convention (p_n ∘ ... ∘ p_2 ∘ p_1).

Sequence: 3 1 4 2 5

Question: What is the single integer ID of the final composed permutation? Your response must be only the integer.

Answer:
```

**Expected Output:** A single integer representing the composed permutation ID.

### Metrics

Each task reports 100 metrics corresponding to sequence lengths from 5 to 500 (in increments of 5):
- Metric names: `"5"`, `"10"`, `"15"`, ..., `"500"`
- Values: Accuracy (0.0-1.0) for samples at that sequence length, or -1 if no samples exist for that length
- Aggregation: Arithmetic mean across all samples at each sequence length

The metrics allow for fine-grained analysis of model performance as sequence length increases, revealing the point at which models fail to maintain accurate state tracking.

### Expected Results

Based on the theoretical analysis in the paper:

1. **State-Space Models (SSMs)** without explicit state-tracking mechanisms are expected to fail (near 0% accuracy) as they cannot maintain the necessary state information for permutation composition.

2. **Transformer-based models** may show varying performance:
   - Models without special training: Typically achieve near 0% accuracy
   - Chain-of-thought approaches: May show improved performance by maintaining intermediate states
   - Performance typically degrades as sequence length increases

3. **Complexity class differences**:
   - TC⁰ groups (solvable) may be slightly easier for some architectures
   - NC¹ groups (non-solvable) represent a harder challenge requiring more complex reasoning

### Dataset Details

The benchmark uses the `BeeGass/Group-Theory-Collection` dataset on Hugging Face, which provides:
- Pre-computed permutation sequences for all 94 groups
- Correct composition results verified against group multiplication tables
- Balanced sampling across different sequence lengths
- Each group's dataset can be independently verified using standard group theory algorithms

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? 
    * The paper provides the theoretical framework. The `BeeGass/Group-Theory-Collection` dataset is a concrete implementation following the paper's methodology. Correctness can be verified via standard group multiplication tables.

### Implementation Notes

1. **Output Type**: All tasks use `loglikelihood` output type for consistent evaluation
2. **Data Loading**: Each group has its own dataset loading function in `group_composition_utils.py`
3. **Metric Processing**: Custom metric processing handles sequence length bucketing and aggregation
4. **Memory Efficiency**: Tasks load data on-demand to handle the large number of groups efficiently

### Changelog

- **v1.0** (2024): Initial release with 94 permutation groups and comprehensive test coverage