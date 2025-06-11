# LiveCodeBench Easy Difficulty Filtering

This document explains how to efficiently filter LiveCodeBench to only evaluate on "easy" difficulty questions, avoiding the need to parse the entire dataset.

## Problem

The original approach in `LiveCodeBenchEasy` class was inefficient because it:
1. Loaded the entire LiveCodeBench dataset into memory
2. Converted all problems to documents
3. Then filtered them by difficulty

This meant parsing thousands of problems even when you only wanted the easy ones.

## Efficient Solution

We've implemented an efficient dataset-level filtering approach using HuggingFace datasets' native filtering capabilities.

### Key Components

1. **`filter_dataset_by_difficulty()`** - Filters at the HuggingFace dataset level before document conversion
2. **`LiveCodeBenchEasyEfficient`** - A task class that uses dataset-level filtering
3. **`livecodebench_easy.yaml`** - Configuration file for the efficient easy-only task

### Usage

#### Option 1: Use the Easy-Only Task (Recommended)

```bash
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks livecodebench_easy \
    --device cuda:0 \
    --batch_size 8
```

This will automatically filter to only easy problems without loading the entire dataset.

#### Option 2: Use the Filtering Function Directly

```python
from datasets import load_dataset
from lm_eval.tasks.livecodebench.utils import filter_dataset_by_difficulty

# Load dataset
dataset = load_dataset("livecodebench/code_generation_lite", 
                      split="test", 
                      version_tag="release_v6")

# Filter to easy problems only
easy_dataset = filter_dataset_by_difficulty(dataset, ["easy"])

print(f"Original: {len(dataset)} problems")
print(f"Easy only: {len(easy_dataset)} problems")
```

#### Option 3: Filter Multiple Difficulties

```python
# Filter for both easy and medium
easy_medium_dataset = filter_dataset_by_difficulty(dataset, ["easy", "medium"])
```

### Performance Comparison

| Approach | Dataset Loading | Memory Usage | Time to First Problem |
|----------|----------------|--------------|---------------------|
| Original `LiveCodeBenchEasy` | All problems | High | Slow |
| New `LiveCodeBenchEasyEfficient` | Easy problems only | Low | Fast |

### Technical Details

The efficient filtering works by:

1. **HuggingFace Dataset Filter**: Uses `dataset.filter()` which is optimized for performance
2. **Lazy Loading**: Only loads problems that pass the filter
3. **Early Termination**: Stops processing as soon as non-matching problems are identified
4. **Memory Efficiency**: Filtered dataset uses significantly less memory

### Testing

Run the test script to verify filtering works correctly:

```bash
cd lm-evaluation-harness
python test_easy_filtering.py
```

### Difficulty Mapping

The filtering supports:

- **Direct mapping**: `"easy"`, `"medium"`, `"hard"`
- **AtCoder contest types**: 
  - `abc*` → `easy`
  - `arc*` → `medium`
  - `agc*` → `hard`

### Migration Guide

If you're currently using the old approach:

**Before:**
```python
# This loads ALL problems first
class LiveCodeBenchEasy:
    def eval_docs(self):
        all_docs = list(self._base_task.eval_docs())  # Inefficient!
        return [doc for doc in all_docs if doc.get('difficulty') == 'easy']
```

**After:**
```python
# This filters at dataset level
class LiveCodeBenchEasyEfficient:
    def download(self, *args, **kwargs):
        dataset = self._base_task.download(*args, **kwargs)
        return filter_dataset_by_difficulty(dataset, ["easy"])  # Efficient!
```

### Files

- `utils.py` - Contains filtering functions and task classes
- `livecodebench_easy.yaml` - Task configuration for easy-only evaluation
- `test_easy_filtering.py` - Test script to verify filtering works
- `README_EASY_FILTERING.md` - This documentation

### Configuration

The `livecodebench_easy.yaml` configuration:

```yaml
task: livecodebench_easy
class: !function utils.LiveCodeBenchEasyEfficient
dataset_path: livecodebench/code_generation_lite
dataset_kwargs:
  version_tag: "release_v6"
# ... rest of configuration
```

This approach provides significant performance improvements when you only need to evaluate on easy problems, making your evaluation runs much faster and more memory-efficient. 