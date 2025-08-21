# Long Context Evaluation Benchmarks

This document provides an overview of the long-context evaluation benchmarks available in lm-evaluation-harness.

## Available Benchmarks

### 1. LongBench v1 (Existing)
- **Path**: `lm_eval/tasks/longbench/`
- **Context Length**: 5k-15k tokens
- **Tasks**: 21 tasks across 6 categories
- **Usage**: `lm_eval --tasks longbench --model <model>`

### 2. LongBench v2 (New)
- **Path**: `lm_eval/tasks/longbench_v2/`
- **Context Length**: 8k-2M tokens
- **Tasks**: 20 tasks with deeper reasoning requirements
- **Usage**: `lm_eval --tasks longbench_v2 --model <model>`
- **Key Features**: Extended context lengths, more challenging reasoning tasks

### 3. Babilong (New)
- **Path**: `lm_eval/tasks/babilong/`
- **Context Length**: 1k-10M tokens
- **Tasks**: 20 reasoning tasks (qa1-qa20)
- **Usage**: `lm_eval --tasks babilong --model <model>`
- **Key Features**: Ultra-long contexts, multi-hop reasoning, haystack design

### 4. InfiniteBench (New)
- **Path**: `lm_eval/tasks/infinitebench/`
- **Context Length**: 100k-1M+ tokens
- **Tasks**: 12 tasks across retrieval, math, code, QA, and dialogue
- **Usage**: `lm_eval --tasks infinitebench --model <model>`
- **Key Features**: Extreme context lengths, diverse task types

### 5. Phonebook Lookup (New)
- **Path**: `lm_eval/tasks/phonebook/`
- **Context Length**: 1k-200k tokens
- **Tasks**: 7 variants with different context lengths
- **Usage**: `lm_eval --tasks phonebook --model <model>`
- **Key Features**: Position sensitivity analysis, retrieval accuracy

### 6. RULER (Existing)
- **Path**: `lm_eval/tasks/ruler/`
- **Context Length**: Variable
- **Usage**: Requires custom arguments (see RULER README)

### 7. LIBRA (Existing - Russian)
- **Path**: `lm_eval/tasks/libra/`
- **Context Length**: 4k-128k tokens
- **Tasks**: 21 tasks including Russian Babilong variants
- **Usage**: `lm_eval --tasks libra --model <model>`

## Quick Start Examples

### Evaluate on all new long-context benchmarks:
```bash
lm_eval --model hf \
  --model_args pretrained=<model_name> \
  --tasks longbench_v2,babilong,infinitebench,phonebook \
  --batch_size 1 \
  --output_path results/
```

### Evaluate on specific context length ranges:

#### Short contexts (1k-10k):
```bash
lm_eval --model hf \
  --model_args pretrained=<model_name> \
  --tasks phonebook_1k,phonebook_5k,phonebook_10k \
  --batch_size 1
```

#### Medium contexts (10k-100k):
```bash
lm_eval --model hf \
  --model_args pretrained=<model_name> \
  --tasks phonebook_25k,phonebook_50k,phonebook_100k \
  --batch_size 1
```

#### Long contexts (100k+):
```bash
lm_eval --model hf \
  --model_args pretrained=<model_name> \
  --tasks infinitebench,phonebook_200k \
  --batch_size 1
```

## Benchmark Comparison

| Benchmark | Min Context | Max Context | # Tasks | Languages | Focus Area |
|-----------|------------|-------------|---------|-----------|------------|
| LongBench v1 | 5k | 15k | 21 | EN, ZH | General |
| LongBench v2 | 8k | 2M | 20 | EN, Multi | Deep reasoning |
| Babilong | 1k | 10M | 20 | EN | Multi-hop reasoning |
| InfiniteBench | 100k | 1M+ | 12 | EN, ZH | Extreme length |
| Phonebook | 1k | 200k | 7 | EN | Position sensitivity |
| RULER | Variable | Variable | Multiple | EN | Synthetic tasks |
| LIBRA | 4k | 128k | 21 | RU | Russian language |

## Implementation Notes

### Memory Requirements
- Tasks with 100k+ tokens require significant GPU memory
- Recommended: 80GB+ GPU memory for 200k+ token tasks
- Use gradient checkpointing and lower batch sizes for memory efficiency

### Dataset Access
Most datasets are hosted on Hugging Face Hub:
- LongBench v2: `THUDM/LongBench-v2`
- Babilong: `booydar/babilong`
- InfiniteBench: `OpenBMB/InfiniteBench`
- Phonebook: `nelson-liu/lost-in-the-middle`

### Custom Configuration
Each benchmark supports custom configuration through YAML files. See individual benchmark directories for examples.

## Contributing

To add new long-context benchmarks:
1. Create a new directory under `lm_eval/tasks/`
2. Include README.md with paper citation and task description
3. Implement utils.py with evaluation metrics
4. Create YAML configuration files for each task
5. Add group configuration file for running all tasks

## Citations

Please cite the original papers when using these benchmarks:

- **LongBench v2**: Bai et al., 2024
- **Babilong**: Kuratov et al., 2024
- **InfiniteBench**: Zhang et al., 2024
- **Phonebook**: Liu et al., 2023

See individual benchmark READMEs for complete citations.