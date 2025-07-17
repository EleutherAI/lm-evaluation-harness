# Molecular Reasoning Evaluation Suite

This directory contains evaluation configurations for molecular reasoning tasks following the lm-eval-harness format.

## File Structure

```
eval_configs/
├── molecular_reasoning.yaml      # Main group configuration
├── molecular_structure_elucidation.yaml
├── molecular_name_conversion.yaml
├── molecular_dou_calculation.yaml
├── molecular_multiple_choice.yaml
├── answer_utils.py              # Answer extraction and matching utilities
└── utils.py                     # Document processing utilities
```

## Main Configuration

- `molecular_reasoning.yaml`: Aggregates all molecular reasoning tasks

## Individual Task Configurations

### 1. Structure Elucidation (`molecular_structure_elucidation.yaml`)
- **Dataset**: `data/structure_elucidation.jsonl`
- **Task**: Build molecules step-by-step based on NMR data and fragment pools
- **Format**: Free-form generation with structured output
- **Key Features**:
  - Max tokens: 1024
  - Stop sequences: "\n\n", "Task:", "</s>", "<|im_end|>"
  - Metrics: Exact match

### 2. Name Conversion (`molecular_name_conversion.yaml`)
- **Dataset**: `data/name_conversion_qa_pairs.jsonl`
- **Task**: Convert SMILES notation to IUPAC names or molecular formulas
- **Format**: Question-answer pairs
- **Key Features**:
  - Extracts answer from "The answer is: X" format
  - Max tokens: 256
  - Metrics: Exact match (case-insensitive)

### 3. Degree of Unsaturation (`molecular_dou_calculation.yaml`)
- **Dataset**: `data/dou_qa_pairs.jsonl`
- **Task**: Calculate degree of unsaturation or determine if molecule is saturated
- **Format**: Question-answer pairs
- **Key Features**:
  - Extracts answer from "The answer is: X" format
  - Max tokens: 128
  - Metrics: Exact match (case and punctuation insensitive)

### 4. Multiple Choice (`molecular_multiple_choice.yaml`)
- **Dataset**: `data/multiple-choice.csv`
- **Task**: Answer multiple choice questions about molecular properties
- **Format**: Standard A/B/C/D multiple choice
- **Key Features**:
  - Extracts single letter answer
  - Max tokens: 10
  - Metrics: Exact match

## Configuration Format

Each task configuration follows this structure:

```yaml
dataset_path: <path_to_dataset>
dataset_name: null
validation_split: null
test_split: train
fewshot_split: train
process_docs: null  # or function reference
doc_to_text: <template_for_input>
doc_to_target: <field_or_template_for_target>
task: <task_name>
filter_list:
  - name: <filter_name>
    filter:
      - function: <filter_function>
        <filter_params>
generation_kwargs:
  max_tokens: <max_tokens>
  until: <stop_sequences>
  do_sample: false
  temperature: 0.0
num_fewshot: 0
metric_list:
  - metric: <metric_name>
    aggregation: mean
    higher_is_better: true
    <other_metric_params>
```



## Utility Files

### `answer_utils.py`
Provides functions for:
- Extracting answers from various formats (XML tags, regex patterns)
- SMILES validation and canonicalization
- Molecular similarity calculations
- Answer normalization and matching

### `utils.py`
Provides functions for:
- Document preprocessing
- Filter functions for answer extraction
- Text normalization
- Format conversion

## Adding New Tasks

1. Create a new YAML configuration following the format above
2. Add the task name to `molecular_reasoning.yaml`
3. If needed, add custom processing functions to `utils.py`
4. Use appropriate filters for answer extraction 