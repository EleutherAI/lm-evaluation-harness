# MMLU-CF

### Paper

Title: MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark

Paper (ACL 2025 Main): https://aclanthology.org/2025.acl-long.656/

MMLU-CF is a contamination-free variant of the MMLU benchmark designed to address data leakage issues in evaluating large language models. The benchmark applies decontamination rules to prevent models from memorizing and reproducing identical answer choices from training data.

Homepage: https://huggingface.co/datasets/microsoft/MMLU-CF

### Citation

```bibtex
@misc{zhao2024mmlucfcontaminationfreemultitasklanguage,
    title={MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark},
    author={Qihao Zhao and Yangyu Huang and Tengchao Lv and Lei Cui and Qinzheng Sun and Shaoguang Mao and Xin Zhang and Ying Xin and Qiufeng Yin and Scarlett Li and Furu Wei},
    year={2024},
    eprint={2412.15194},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2412.15194},
}
```

### Groups and Tasks

#### Groups

* `mmlu_cf`: All 15 subjects
  * `mmlu_cf_stem`: STEM subjects (Biology, Math, Chemistry, Physics, Engineering, Computer Science)
  * `mmlu_cf_social_sciences`: Social science subjects (Economics, Psychology, Business, Law, History)
  * `mmlu_cf_humanities`: Humanities subjects (Philosophy)
  * `mmlu_cf_other`: Other subjects (Health, Other)

#### Tasks

| Task | Subject | Category |
|------|---------|----------|
| `mmlu_cf_biology` | Biology | STEM |
| `mmlu_cf_math` | Math | STEM |
| `mmlu_cf_chemistry` | Chemistry | STEM |
| `mmlu_cf_physics` | Physics | STEM |
| `mmlu_cf_engineering` | Engineering | STEM |
| `mmlu_cf_computer_science` | Computer Science | STEM |
| `mmlu_cf_economics` | Economics | Social Sciences |
| `mmlu_cf_psychology` | Psychology | Social Sciences |
| `mmlu_cf_business` | Business | Social Sciences |
| `mmlu_cf_law` | Law | Social Sciences |
| `mmlu_cf_history` | History | Social Sciences |
| `mmlu_cf_philosophy` | Philosophy | Humanities |
| `mmlu_cf_health` | Health | Other |
| `mmlu_cf_miscellaneous` | Other/Miscellaneous | Other |

### Dataset Structure

The dataset contains:
- **Validation set**: 10,000 questions (open-source, used as test split)
- **Development set**: 5 questions per subject (used for few-shot examples)
- **Test set**: 10,000 questions (closed-source, not available)

Each example has the following fields:
- `Question`: The question text
- `A`, `B`, `C`, `D`: Four answer choices
- `Answer`: The correct answer (A, B, C, or D)

### Usage

Run all MMLU-CF tasks:
```bash
lm-eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks mmlu_cf \
    --batch_size auto
```

Run specific category:
```bash
lm-eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks mmlu_cf_stem \
    --batch_size auto
```

Run individual task:
```bash
lm-eval --model hf \
    --model_args pretrained=<model_name> \
    --tasks mmlu_cf_biology \
    --batch_size auto
```

### Differences from Original MMLU

| Aspect | MMLU | MMLU-CF |
|--------|------|---------|
| Contamination Prevention | Not addressed | Applies 3 decontamination rules |
| Test Set | Open-source | Closed-source (prevents leakage) |
| Data Leakage Risk | Higher | Mitigated through decontamination |
| Quality Validation | Basic | Reviewed by GPT-4o, Gemini, Claude |

### License

CDLA-2.0 (Community Data License Agreement - Permissive 2.0)
