# KenPOS: Kenyan Languages Part-of-Speech Tagging

## 📋 Overview

Evaluation task for Part-of-Speech (POS) tagging on low-resource Kenyan languages using the **`generate_until`** function from lm-evaluation-harness. This task uses text generation (not multiple choice) to evaluate language models' ability to predict POS tags for words in Dholuo and Luhya dialects.

### Dataset Information

| Language | Code | Family | Tagged Words | Description |
|----------|------|--------|--------------|-------------|
| Dholuo | `dho` | Nilotic | ~50,000 | Spoken by ~4.2M people in Kenya |
| Luhya-Lumarachi | `lch` | Bantu (Luhya) | ~27,900 | Luhya dialect from Western Kenya |
| Luhya-Lulogooli | `llg` | Bantu (Luhya) | ~34,300 | Luhya dialect (also Logooli) |
| Luhya-Lubukusu | `lbk` | Bantu (Luhya) | ~30,900 | Luhya dialect (also Bukusu) |

**Total:** ~143,000 tagged words across 4 languages

**Source:** [Kencorpus/KenPOS](https://huggingface.co/datasets/Kencorpus/KenPOS) on HuggingFace

---

## 📁 Project Structure

```
lm_eval/tasks/kenpos/
├── README.md                    # This file (documentation)
├── utils.py                     # Processing functions (you create manually)
├── _generate_configs.py         # YAML generator script (you create manually)
├── _default_template_yaml       # Base config template (you create manually)
├── dho.yaml                     # Dholuo config (auto-generated)
├── lch.yaml                     # Lumarachi config (auto-generated)
├── llg.yaml                     # Lulogooli config (auto-generated)
└── lbk.yaml                     # Lubukusu config (auto-generated)
```

**Key Files:**
- **Manual files (3):** You create these once
  - `utils.py` - Python functions for data processing
  - `_default_template_yaml` - Base YAML configuration
  - `_generate_configs.py` - Script to generate language YAMLs
- **Generated files (4):** Created automatically by running `_generate_configs.py`
  - `dho.yaml`, `lch.yaml`, `llg.yaml`, `lbk.yaml`

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .

# Install dependencies
pip install datasets huggingface_hub pyyaml "numpy<2.0"
```

### Step 1: Create Task Directory

```bash
cd lm_eval/tasks
mkdir kenpos
cd kenpos
```

### Step 2: Create Manual Files

Create these 3 files (see **File Contents** section below):
1. `utils.py`
2. `_default_template_yaml`
3. `_generate_configs.py`

### Step 3: Generate YAML Configs

```bash
python _generate_configs.py
```

**Expected output:**
```
Generating KenPOS configs...
  ✓ dho.yaml
  ✓ lch.yaml
  ✓ llg.yaml
  ✓ lbk.yaml

Done!
```

### Step 4: Verify Tasks

```bash
cd ../../..  # Back to lm-eval root
python -m lm_eval --tasks list | grep kenpos
```

**Expected output:**
```
kenpos_dho
kenpos_lch
kenpos_llg
kenpos_lbk
```

### Step 5: Run Quick Test

```bash
python -m lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks kenpos_dho \
    --device cpu \
    --limit 5
```

---

## 🎯 Evaluation Methods

### 1. Quick Test (Limited Samples)

Test with just 5 samples to verify everything works:

```bash
python -m lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks kenpos_dho \
    --device cpu \
    --limit 5
```

**Use case:** Verify setup before full evaluation  
**Time:** ~30 seconds  
**Note:** Metrics not accurate with `--limit`

---

### 2. Single Language Evaluation

Evaluate on one language:

```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks kenpos_dho \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/kenpos_dho/
```

**Parameters:**
- `--tasks kenpos_dho` - Evaluate Dholuo only
- `--device cuda:0` - Use GPU (or `cpu` if no GPU)
- `--batch_size 8` - Process 8 examples at once
- `--output_path` - Where to save results

**Outputs:**
```
results/kenpos_dho/
├── results.json              # Aggregated metrics
└── samples_kenpos_dho.jsonl  # Individual predictions (with --log_samples)
```

---

### 3. Multiple Languages Evaluation

Evaluate on specific languages:

```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks kenpos_dho,kenpos_lch,kenpos_llg,kenpos_lbk \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/kenpos_all/ \
    --log_samples
```

**Note:** Comma-separated task names, no spaces!

**Outputs:**
```
results/kenpos_all/
├── results.json                 # All results aggregated
├── samples_kenpos_dho.jsonl    # Dholuo predictions
├── samples_kenpos_lch.jsonl    # Lumarachi predictions
├── samples_kenpos_llg.jsonl    # Lulogooli predictions
└── samples_kenpos_lbk.jsonl    # Lubukusu predictions
```

---

### 4. Few-Shot Evaluation

Provide examples before each test (in-context learning):

```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks kenpos_dho \
    --num_fewshot 3 \
    --device cuda:0 \
    --output_path results/fewshot_3/
```

**Few-shot options:**
- `--num_fewshot 0` - Zero-shot (no examples)
- `--num_fewshot 1` - One example before each test
- `--num_fewshot 3` - Three examples (recommended)
- `--num_fewshot 5` - Five examples

**Example prompt with 3-shot:**
```
Tag the following word with its part of speech.

Word: nyathi
POS Tag: NOUN

Word: gi
POS Tag: ADP

Word: chiemo
POS Tag: NOUN

Word: [test_word]
POS Tag:
```

---

### 5. Save Individual Predictions

Save every single prediction for detailed analysis:

```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks kenpos_dho \
    --device cuda:0 \
    --output_path results/with_samples/ \
    --log_samples
```

**Sample output format (`samples_kenpos_dho.jsonl`):**
```json
{"doc_id": 0, "doc": {"token": "AUSTINE", "pos_tag": "NN"}, "target": " NN", "resps": [[" the"]], "filtered_resps": [" the"], "exact_match": 0.0}
{"doc_id": 1, "doc": {"token": "OCHIENG", "pos_tag": "NN"}, "target": " NN", "resps": [[" O"]], "filtered_resps": [" O"], "exact_match": 0.0}
```

---

### 6. CPU vs GPU Evaluation

**CPU (slower, no GPU required):**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks kenpos_dho \
    --device cpu \
    --batch_size 2
```

**GPU (faster, requires CUDA):**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks kenpos_dho \
    --device cuda:0 \
    --batch_size 16
```

**Multiple GPUs:**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model-name,parallelize=True \
    --tasks kenpos_dho \
    --batch_size 32
```

---

### 7. Different Model Sources

**HuggingFace Hub:**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks kenpos_dho
```

**Local Model:**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=/path/to/your/model \
    --tasks kenpos_dho
```

**With Specific Revision:**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=your-model,revision=main \
    --tasks kenpos_dho
```

---

### 8. Batch Processing Multiple Models

Create a script to evaluate multiple models:

```bash
#!/bin/bash
# evaluate_all_models.sh

MODELS=(
    "gpt2"
    "gpt2-medium"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-410m"
)

for MODEL in "${MODELS[@]}"; do
    echo "Evaluating $MODEL"
    python -m lm_eval --model hf \
        --model_args pretrained=$MODEL \
        --tasks kenpos_dho,kenpos_lch,kenpos_llg,kenpos_lbk \
        --device cuda:0 \
        --batch_size 8 \
        --output_path results/$(basename $MODEL)/ \
        --log_samples
done

echo "All evaluations complete!"
```

---

### 9. Advanced Options

**Verbose logging:**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks kenpos_dho \
    --verbosity DEBUG
```

**Custom generation parameters:**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks kenpos_dho \
    --gen_kwargs temperature=0.1,top_p=0.9
```

**Auto batch size (finds optimal size):**
```bash
python -m lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks kenpos_dho \
    --batch_size auto
```

---

## 📊 Understanding Results

### Results JSON Structure

**Location:** `results/kenpos_dho/results.json`

```json
{
  "results": {
    "kenpos_dho": {
      "exact_match": 0.234,
      "alias": "kenpos_dho"
    }
  },
  "configs": {
    "kenpos_dho": {
      "task": "kenpos_dho",
      "dataset_path": "Kencorpus/KenPOS",
      "dataset_name": "dho",
      "output_type": "generate_until"
    }
  },
  "versions": {
    "kenpos_dho": 1
  }
}
```

### Interpreting Scores

- **exact_match: 0.234** = 23.4% of predictions exactly matched ground truth
- **exact_match: 1.0** = 100% accuracy (perfect)
- **exact_match: 0.0** = 0% accuracy (no correct predictions)

### Sample Predictions Analysis

```python
import json

# Load samples
with open('results/kenpos_dho/samples_kenpos_dho.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

# Analyze predictions
correct = sum(1 for s in samples if s['exact_match'] == 1.0)
total = len(samples)
print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

# See mistakes
mistakes = [s for s in samples if s['exact_match'] == 0.0]
for m in mistakes[:5]:
    print(f"Word: {m['doc']['token']}")
    print(f"  Predicted: {m['filtered_resps'][0]}")
    print(f"  Actual: {m['doc']['pos_tag']}")
    print()
```

---

## ⚙️ How It Works

### Evaluation Pipeline

```
1. Load Dataset
   └─> HuggingFace: Kencorpus/KenPOS (config: dho)
   
2. For each example:
   └─> doc_to_text() creates prompt
       Input: {"token": "nyathi", "pos_tag": "NOUN"}
       Output: "Tag the following word...\nWord: nyathi\nPOS Tag:"
   
3. Model generates response
   └─> generate_until() function
       Generates until: \n, ., or 10 tokens
       Example output: " NOUN"
   
4. Process results
   └─> process_results() compares
       Predicted: "NOUN"
       Actual: "NOUN"
       Match: ✓ (exact_match = 1.0)
   
5. Aggregate metrics
   └─> Mean of all exact_match scores
```

### Generation Settings

```yaml
output_type: generate_until  # Text generation (not multiple choice)

generation_kwargs:
  until: ["\n", ".", ","]    # Stop at newline, period, or comma
  max_gen_toks: 10           # Maximum 10 tokens
  do_sample: false           # Greedy decoding (no randomness)
  temperature: 0.0           # No temperature sampling
```

---

## Troubleshooting

### Issue: Tasks Not Found

```bash
# Error: Tasks not found: kenpos_dho
```

**Solution:**
```bash
# Make sure you're in the lm-eval root directory
cd /path/to/lm-evaluation-harness

# Verify files exist
ls lm_eval/tasks/kenpos/

# Should see: dho.yaml, lch.yaml, llg.yaml, lbk.yaml
```

---

### Issue: NumPy Version Error

```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

**Solution:**
```bash
pip install "numpy<2.0"
```

---

### Issue: CUDA Not Available

```
AssertionError: Torch not compiled with CUDA enabled
```

**Solution:**
```bash
# Use CPU instead
python -m lm_eval --tasks kenpos_dho --device cpu ...
```

---

### Issue: Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce batch size
--batch_size 1

# Use CPU
--device cpu

# Use smaller model
--model_args pretrained=gpt2  # instead of gpt2-large
```

---

## 📚 File Contents

### Complete `utils.py`

```python
"""
KenPOS Dataset Utilities
Helper functions for processing KenPOS POS tagging data
"""


def doc_to_text(doc):
    """Convert document to input text for the model"""
    word = doc.get("token", "")
    
    prompt = f"""Tag the following word with its part of speech.

Word: {word}
POS Tag:"""
    
    return prompt


def doc_to_target(doc):
    """Extract the target POS tag from the document"""
    pos_tag = doc.get("pos_tag", "")
    return f" {pos_tag}"


def process_results(doc, results):
    """Process the model's generated results and calculate metrics"""
    pred = results[0].strip() if results else ""
    pred_tag = pred.split()[0] if pred.split() else pred
    target = doc.get("pos_tag", "")
    
    # Return only numeric metric (not strings!)
    exact_match = 1.0 if pred_tag.upper() == target.upper() else 0.0
    
    return {
        "exact_match": exact_match
    }


def doc_to_decontamination_query(doc):
    """Generate decontamination query"""
    word = doc.get("token", "")
    return word
```

---

### Complete `_default_template_yaml`

```yaml
# KenPOS Base Template Configuration
output_type: generate_until

# Dataset configuration - only 'train' split exists
dataset_path: Kencorpus/KenPOS
training_split: null
validation_split: train
test_split: train

# Document processing functions from utils.py
doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target
process_results: !function utils.process_results

# Decontamination settings
should_decontaminate: true
doc_to_decontamination_query: !function utils.doc_to_decontamination_query

# Generation parameters for generate_until function
generation_kwargs:
  until:
    - "\n"
    - "."
    - ","
  max_gen_toks: 10
  do_sample: false
  temperature: 0.0

# Metrics
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true

# Metadata
metadata:
  version: 1.0
  description: "Part-of-Speech tagging for Kenyan languages using generate_until"
  reference: "https://arxiv.org/abs/2208.12081"
```

---

### Complete `_generate_configs.py`

```python
#!/usr/bin/env python3
"""Generate KenPOS YAML configs"""
import yaml

def generate_kenpos_configs():
    print("Generating KenPOS configs...")
    
    languages = {
        "dho": {"full_name": "Dholuo", "family": "Nilotic", "words": 50000},
        "lch": {"full_name": "Lumarachi", "family": "Bantu", "words": 27900},
        "llg": {"full_name": "Lulogooli", "family": "Bantu", "words": 34300},
        "lbk": {"full_name": "Lubukusu", "family": "Bantu", "words": 30900}
    }
    
    for code, info in languages.items():
        config = {
            "include": "_default_template_yaml",
            "task": f"kenpos_{code}",
            "dataset_name": code,
            "metadata": {
                "version": 1.0,
                "language": code,
                "full_name": info["full_name"],
                "description": f"POS tagging for {info['full_name']} (~{info['words']:,} words)"
            }
        }
        
        with open(f"{code}.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ {code}.yaml")
    
    print("\nDone!")

if __name__ == "__main__":
    generate_kenpos_configs()
```

---

## Citation

```bibtex
@article{wanjawa2022kencorpus,
  title={Kencorpus: A Kenyan Language Corpus of Swahili, Dholuo and Luhya for Natural Language Processing Tasks},
  author={Wanjawa, Barack and Indede, Florence and Muchemi, Lawrence and Wanzare, Lilian DA and Ombui, Edward and McOnyango, Owen},
  journal={arXiv preprint arXiv:2208.12081},
  year={2022}
}
```

---

## 🔗 Resources

- **Dataset:** https://huggingface.co/datasets/Kencorpus/KenPOS
- **Paper:** https://arxiv.org/abs/2208.12081
- **LM-Eval Harness:** https://github.com/EleutherAI/lm-evaluation-harness
- **Documentation:** https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs

---

## Checklist

- [ ] Install lm-evaluation-harness
- [ ] Create `kenpos/` directory
- [ ] Create `utils.py`
- [ ] Create `_default_template_yaml`
- [ ] Create `_generate_configs.py`
- [ ] Run `python _generate_configs.py`
- [ ] Verify with `lm_eval --tasks list | grep kenpos`
- [ ] Run quick test with `--limit 5`
- [ ] Run full evaluation with `--output_path`
- [ ] Analyze results

---

**Version:** 1.0  
**Last Updated:** November 2025  
**Status:** Working and tested