# IndicParam

Multiple-choice benchmark covering 13,207 questions across 12 subsets of low-resource Indic languages from Indian competitive exam papers (UGC-NET).

## Dataset

- **HuggingFace**: [bharatgenai/IndicParam](https://huggingface.co/datasets/bharatgenai/IndicParam)
- **Paper**: [IndicParam: Benchmarking LLMs on Low-Resource Indic Languages](https://arxiv.org/abs/2512.00333)
- **Reference impl**: https://github.com/ayushbits/IndicParam
- **Split**: test only (13,207 examples)
- **Config**: `IndicParam` (single config, filtered by `subject` column)

## Languages / Subtasks

| Task name | Subject (dataset value) |
|---|---|
| `indicparam_bodo` | Bodo |
| `indicparam_dogri` | Dogri |
| `indicparam_gujarati` | Gujarati_surya |
| `indicparam_konkani` | Konkani |
| `indicparam_maithili` | Maithili |
| `indicparam_marathi` | Marathi |
| `indicparam_nepali` | Nepali |
| `indicparam_oriya` | Oriya |
| `indicparam_rajasthani` | Rajasthani |
| `indicparam_sanskrit` | Sanskrit |
| `indicparam_sanskrit_mix` | Sanskrit Mix |
| `indicparam_santali` | Santali |

## Task format

- `output_type`: `multiple_choice`
- Prompt: `Question: {question_text}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:`
- Target: index of correct choice (from `correct_answer` column: `a/b/c/d`)
- Metric: `acc` (mean)

## Usage

```bash
# All languages
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks indicparam

# Single language
lm_eval --model hf --model_args pretrained=meta-llama/Llama-2-7b-hf --tasks indicparam_gujarati
```

## Citation

```bibtex
@misc{joshi2025indicparam,
  title={IndicParam: Benchmarking Large Language Models on Low-Resource Indic Languages},
  author={Joshi, Ayush and others},
  year={2025},
  eprint={2512.00333},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.00333}
}
```
