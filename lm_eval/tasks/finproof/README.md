# FinProof BFSI — Adversarial Benchmark for Financial AI Safety

**Source:** [Zytra/finproof-bench](https://huggingface.co/datasets/Zytra/finproof-bench)  
**Paper:** [SSRN 7464798](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7464798)  
**Maintainer:** SreeRamaKrishna Mikkilineni, Zytra Techsolutions (msr.krishna@zytratechnologies.com)

---

## Overview

FinProof is the first adversarial benchmark for evaluating AI guardrails in Banking, Financial Services and Insurance (BFSI). It measures whether a language model (or a guarded system) correctly refuses adversarial financial inputs while remaining helpful on benign queries.

The public Tier-1 subset (1,228 rows) contains:
- **782** benign English queries (B-series)  
- **175** benign English v2 queries  
- **112** Hindi adversarial attack prompts (T1 public)  
- **44** English IICL (Involuntary In-Context Learning) seed attack prompts (T1 public)  
- **30** benign D-series (agentic) queries  
- **85** benign B-series v2 queries

## Tasks

| Task name | Config | n | Content |
|---|---|---|---|
| `finproof_bfsi` | `default` | 1,228 | Full T1: benign + Hindi attacks |
| `finproof_bfsi_iicl` | `iicl_seeds` | 44 | English IICL seed attacks only |

## Metrics

| Metric | Split | Higher is better | Description |
|---|---|---|---|
| `refusal_rate` | attacks | ✓ | Fraction of attack prompts correctly refused |
| `asr` | attacks | ✗ | Attack Success Rate — fraction model complied with |
| `helpfulness_rate` | benign | ✓ | Fraction of benign queries answered (not over-refused) |
| `fpr` | benign | ✗ | False Positive Rate — fraction of benign queries incorrectly refused |

A well-calibrated model achieves **high refusal_rate + high helpfulness_rate**. Raw ASR without FPR is misleading — a model that refuses everything scores 0% ASR but is useless.

## Attack Categories (B-series)

| Category | Description | Regulatory anchor |
|---|---|---|
| B-01 | Investment Advice Elicitation | SEBI IA Regulations 2013 §3 |
| B-02 | Insider Trading Facilitation | SEBI PIT Regulations 2015 |
| B-03 | KYC Bypass Facilitation | RBI KYC Master Direction 2023 |
| B-04 | AML / Structuring Assistance | PMLA 2002, FATF Rec.10-16 |
| B-05 | Consumer Deception | RBI Fair Practices Code |
| B-06 | Unauthorized Credit Advice | RBI Credit Information Guidelines |
| B-07 | Privacy Data Extraction | DPDP Act 2023 |

IICL attacks span all 7 B-series categories using annotation-table framing: the adversarial request is embedded inside a plausible compliance dataset annotation task, exploiting the model's in-context instruction-following tendency.

## Running

```bash
# IICL English attacks only (recommended for first run)
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
        --tasks finproof_bfsi_iicl \
        --batch_size 16

# Full T1 (benign + attacks)
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
        --tasks finproof_bfsi \
        --batch_size 16
```

## Citation

```bibtex
@article{zytra2026finproof,
  title   = {FINPROOF: The First Adversarial Benchmark for AI Guardrail Systems
             in Banking, Financial Services, and Insurance},
  author  = {Zytra Techsolutions},
  journal = {SSRN Working Paper},
  number  = {7464798},
  year    = {2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7464798}
}
```

## Tier structure

T1 (this task) is freely downloadable. T2 (email registration) and T3 (research agreement) contain additional attack prompts including Quantum-generated variants. T4 is the blind held-out test set evaluated server-side at finproof.ai. See the [dataset card](https://huggingface.co/datasets/Zytra/finproof-bench) for access.
