# MMLU-Pro-Clean

`mmlu_pro_clean` is a corrected drop-in for `mmlu_pro`: the same benchmark with the **1,343 answer-key-audit-flagged broken items removed** (12,032 → 10,689). An item is removed if it has a wrong answer key, a malformed structure, or more than one defensible answer (Allcock 2026, *When the Answer Key Is Wrong*).

### How it works

The task loads the **same upstream dataset** as `mmlu_pro` (`TIGER-Lab/MMLU-Pro`) and additionally filters out the audit's exclusion set. The removed IDs are shipped here as the official integer `question_id` in `mmlu_pro_clean_exclusions.json` (1,343 ids), so there is **no re-hosted data and no id remapping** — the correction is fully transparent and inspectable. Verified: all 1,343 ids are present upstream and the filter yields exactly 10,689 items.

Everything else — the 5-shot chain-of-thought prompt, the `custom-extract` answer regex, and the `exact_match` metric — is identical to `mmlu_pro`, so scores are directly comparable.

### Usage

```bash
# corrected set
lm_eval --model hf --model_args pretrained=<model> --tasks mmlu_pro_clean

# report BOTH for a like-for-like comparison with published numbers:
lm_eval --model hf --model_args pretrained=<model> --tasks mmlu_pro,mmlu_pro_clean
```

Per-subject tasks (`mmlu_pro_clean_biology`, …, `mmlu_pro_clean_psychology`) are available and aggregate into the `mmlu_pro_clean` group (size-weighted `exact_match`), mirroring `mmlu_pro`.

### Provenance

- Paper: *When the Answer Key Is Wrong: A Calibrated Audit of GPQA, MMLU-Pro, and MMMU-Pro, with Corrected Releases* (Allcock, 2026) — https://github.com/adamallcock/answer-key-audit
- Corrected dataset + full flagged-candidate ledger: https://github.com/adamallcock/mmlu-pro-clean and `adamallcock/mmlu-pro-clean` on the Hub.
- The audit is model-adjudicated and calibrated (100% recall on injected defects; measured false-positive rate on never-flagged controls), not human-certified; per-item confidence tiers accompany the release.

### Files

| file | purpose |
|---|---|
| `_mmlu_pro_clean.yaml` | group over the 14 subject tasks |
| `_default_template_yaml` | shared config (source, prompt, filter, metric) |
| `mmlu_pro_clean_<subject>.yaml` | one per subject; `process_docs` filters subject + exclusions |
| `utils.py` | prompt formatting + the exclusion filter |
| `mmlu_pro_clean_exclusions.json` | 1,343 removed `question_id`s |
