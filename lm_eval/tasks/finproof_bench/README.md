# FinProof Bench

**BFSI Adversarial Guardrail Benchmark** — evaluates AI guardrails and LLMs against 17 attack categories in Banking, Financial Services and Insurance.

## Task Description

FinProof Bench tests whether a model correctly identifies financially-harmful or regulation-violating prompts (attacks) vs. legitimate BFSI queries (benign). Attack = positive class.

Regulatory anchors: RBI KYC Master Directions · SEBI IA Regulations · IRDAI · DPDP Act 2023 · EU AI Act Art. 9/13/52 · FATF Rec. 10-16 · BCBS Basel III.

## Metrics

| Metric | Description |
|---|---|
| `f1` | Macro F1 (attack = positive class) |
| `precision` | Precision on attack class |
| `recall` | Attack detection rate (recall) |
| `fpr` | False positive rate on benign queries |
| `acc` | Accuracy |

## Dataset

- **HuggingFace**: [`Zytra/finproof-bench`](https://huggingface.co/datasets/Zytra/finproof-bench)
- **T1 public**: 1,228 rows (no login required)
- **T2 gated**: 3,005 rows (click "Access repository" on HF — approved ≤ 24 h)
- **Paper**: arXiv:2604.19461

## Usage

```bash
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
    --tasks finproof_bench \
    --device cuda:0 \
    --batch_size 8
```

## Leaderboard (September 2026)

| Model | Organisation | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **AVAL v1.4** | **Zytra** | **0.968** | **0.998** | — |
| Granite Guardian 3.3 | IBM Research | 0.813 | 0.695 | 10.2% |
| ShieldGemma 9B | Google | 0.731 | 0.578 | 2.9% |
| LlamaGuard 3 | Meta AI | 0.569 | 0.397 | 0.0% |
| WildGuard 7B | AllenAI | 0.346 | 0.209 | 0.1% |

Submit results: finproof@zytratechnologies.com

## Citation

```bibtex
@article{zytra2026finproof,
  title   = {FINPROOF: The First Adversarial Benchmark for AI Guardrail
             Systems in Banking, Financial Services, and Insurance},
  author  = {Zytra Techsolutions},
  year    = {2026},
  note    = {finproof.ai · DIPP199187}
}
```

IICL attack technique: arXiv:2604.19461
