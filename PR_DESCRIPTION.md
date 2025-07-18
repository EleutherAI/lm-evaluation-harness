# Add Permutation Composition Benchmark for Group Theory State-Tracking

## Description

This PR adds a comprehensive permutation composition benchmark to lm-evaluation-harness, implementing 94 mathematical groups divided into two computational complexity classes (TC⁰ and NC¹). The benchmark evaluates models' ability to track state through sequential permutation compositions.

## Key Features

- **94 permutation groups** (58 TC⁰ solvable, 36 NC¹ non-solvable)
- **Fine-grained evaluation** with 100 metrics per group (sequence lengths 5-500)
- **Integration with HuggingFace dataset** [`BeeGass/Group-Theory-Collection`](https://huggingface.co/datasets/BeeGass/Group-Theory-Collection)
- **Theoretical grounding** from "The Illusion of State in State-Space Models" by [William Merrill (@viking-sudo-rm)](https://github.com/viking-sudo-rm)

## Changes

- Added 94 individual task YAML files for each group
- Added 3 group aggregation files (permutation_groups, tc0_groups, nc1_groups)
- Added utility module `group_composition_utils.py` for dataset loading and metrics
- Added comprehensive documentation in `README.md`
- Included task generation script for maintainability

## Dataset Resources

- **HuggingFace Dataset**: https://huggingface.co/datasets/BeeGass/Group-Theory-Collection
- **Dataset Generator**: https://github.com/BeeGass/Group-Dataset-Generator

## Testing

- Successfully tested with GPT-2 model
- Validated all tasks load correctly with dummy model
- Pre-commit hooks pass (formatting, linting)

## Usage

```bash
# Evaluate all groups
lm_eval --model hf --model_args pretrained=model_name --tasks permutation_groups

# Evaluate complexity classes
lm_eval --model hf --model_args pretrained=model_name --tasks tc0_groups
lm_eval --model hf --model_args pretrained=model_name --tasks nc1_groups

# Evaluate specific groups
lm_eval --model hf --model_args pretrained=model_name --tasks s3_composition,a5_composition
```

## Benchmark Structure

### TC⁰ Groups (58 total - solvable groups)
- **Symmetric**: S3, S4
- **Alternating**: A3, A4  
- **Cyclic**: C2-C30 (29 groups)
- **Dihedral**: D3-D20 (18 groups)
- **Quaternion**: Q8, Q16, Q32
- **Frobenius**: F20, F21
- **Klein four-group**: V4
- **Elementary abelian**: Z2¹-Z2⁵, Z3¹-Z3⁴, Z5¹-Z5⁴
- **Projective Special Linear**: PSL(2,2), PSL(2,3)

### NC¹ Groups (36 total - non-solvable groups)
- **Symmetric**: S5-S9
- **Alternating**: A5-A9
- **Projective Special Linear**: PSL(2,4/5/7/8/9/11), PSL(3,2/3/4/5)
- **Mathieu**: M11, M12

## Motivation

This benchmark provides a rigorous test of models' state-tracking capabilities through group theory operations, enabling researchers to evaluate whether models can truly maintain state or merely approximate it. The division into TC⁰ and NC¹ complexity classes allows for fine-grained analysis of computational boundaries in neural architectures.