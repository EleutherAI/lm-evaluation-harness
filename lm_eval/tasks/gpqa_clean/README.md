# GPQA-Clean

Corrected drop-in variants of the `gpqa` zero-shot tasks, with the answer-key-audit-flagged broken items removed (wrong key, malformed, or more than one defensible answer; Allcock 2026, *When the Answer Key Is Wrong*):

| task | base | removed | items |
|---|---|---|---|
| `gpqa_diamond_clean_zeroshot` | `gpqa_diamond_zeroshot` | 9 | 198 → 189 |
| `gpqa_extended_clean_zeroshot` | `gpqa_extended_zeroshot` | 48 | 546 → 498 |

### How it works

Each task loads the **same upstream `Idavidrein/gpqa`** as the `gpqa` task and additionally filters out the audit's exclusion set, shipped here as GPQA `Record ID`s (`gpqa_diamond_clean_exclusions.json`, `gpqa_extended_clean_exclusions.json`) — so there is **no re-hosted data and no id remapping**. Everything else (the choice-shuffle preprocessing, the zero-shot prompt, and the `acc`/`acc_norm` metrics) is identical to `gpqa`, so scores are directly comparable.

The `gpqa_extended_clean` set is the union of both audited halves (Diamond + the Extended-minus-Diamond complement) over the full 546-item Extended.

### Usage

```bash
lm_eval --model hf --model_args pretrained=<model> --tasks gpqa_diamond_clean_zeroshot
lm_eval --model hf --model_args pretrained=<model> --tasks gpqa_extended_clean_zeroshot

# report BOTH for a like-for-like comparison with published numbers:
lm_eval --model hf --model_args pretrained=<model> --tasks gpqa_diamond_zeroshot,gpqa_diamond_clean_zeroshot
```

> **Access note:** `Idavidrein/gpqa` is gated on the Hub — accept the GPQA license with your HF account first (same as the base `gpqa` task).

### Provenance

- Paper: *When the Answer Key Is Wrong* (Allcock, 2026) — https://github.com/adamallcock/answer-key-audit
- Corrected datasets + ledgers: `gpqa-diamond-clean`, `gpqa-extended-clean` (github.com/adamallcock).
- Verdicts are model-adjudicated and calibrated, not human-certified; per-item confidence tiers accompany the releases.
