# FANToM — lm-eval adapter

Faithful adaptation of the FANToM benchmark (Kim et al., EMNLP 2023; arXiv:2310.15421)
for `lm-evaluation-harness`. Scoring **vendors FANToM's own** `setup_fantom` /
per-QA scorers / `score_and_analyze` logic from the read-only submodule
(`benchmarks/fantom/`), invoked through lm-eval's `process_results` + custom
`aggregation` hooks so the headline numbers are reported live, in-harness.

## Tasks

| Task | Context | Prompting | Aggregation target |
|------|---------|-----------|--------------------|
| `fantom` | `short` | direct | `set` |
| `fantom_full` | `full` | direct | `set` |
| `fantom_cot` | `short` | zero-shot CoT (single-pass) | `set` |

## Architecture

- **doc = one flattened QA.** FANToM prompts each question separately (a *per-item*
  protocol), so 870 sets explode into **12,832 docs** (`short`). Question types are NOT
  split into separate tasks — the headline ALL\*/ALL scores are set-level conjunctions
  over *every* question type in a set, so they must all be scored together.
- **`output_type: generate_until`**, prompt = the exact string `setup_fantom` builds.
- **`process_results`** scores the QA with the vendored per-type scorers (token-F1 for
  fact; RoBERTa cosine for BeliefQ[Dist.]; letter-match for BeliefQ[Choice]; set
  membership for list; yes/no mapping for binary) and emits a rich per-QA payload.
- **Aggregation** (one custom `!function` per metric) rebuilds the corpus DataFrame from
  the emitted payloads and runs vendored `score_and_analyze` for **both** the
  `inaccessible` (FANToM) and `accessible` (control) scenarios.

## Metrics (23, reported as percentages ×100)

The 12 FANToM headline numbers (the paper's results table) for the main task, plus the
same 12 for the mandated **control** task (`control_*`, the `accessible` scenario; fact
F1 is shared so it appears once).

| Metric | Paper column | Notes |
|--------|--------------|-------|
| `all_star` | All\* | set-level conjunction incl. BeliefQ[Dist.] (the headline) |
| `all` | All | set-level conjunction using BeliefQ[Choice] |
| `belief_choice` | BeliefQ [Choice] | MC letter match |
| `belief_dist` | BeliefQ [Dist.] | embedder: closer to correct than wrong |
| `belief_tokenf1` | BeliefQ token-F1 | token-F1 on the matched view (correct only) |
| `answerability_all` | All AnswerabilityQ | set-level conjunction over answerability |
| `answerability_list` | AnswerabilityQ [List] | set membership |
| `answerability_binary_f1` | AnswerabilityQ [Y/N] | corpus weighted-F1 |
| `infoaccess_all` | All Info-AccessQ | set-level conjunction over info-access |
| `infoaccess_list` | Info-AccessQ [List] | set membership |
| `infoaccess_binary_f1` | Info-AccessQ [Y/N] | corpus weighted-F1 |
| `fact_tokenf1` | FactQ token-F1 | token-F1 |

`control_*` = the same numbers on the `accessible` (no information-asymmetry) subset.
**The paper asks reporters to confirm the control scores stay stable** — check the
`control_*` columns alongside the headline `all_star`/`all`.

## Running

Run inside the harness env (Python 3.13). On Windows set `PYTHONIOENCODING=utf-8` so the
results table prints. The first run downloads the RoBERTa embedder (~1.4 GB) and the
FANToM data (to `~/.cache/fantom`, not committed — eval-only license).

```bash
PYTHONIOENCODING=utf-8 lm-eval --model hf --model_args pretrained=<model> \
    --tasks fantom --include_path lm-evaluation-harness/lm_eval/tasks \
    --limit 30 --log_samples --output_path out/
```

For instruction-tuned/chat models add `--apply_chat_template` (FANToM's own chat agents
apply the chat template; base models do not).

> **`--limit` inflates set-level metrics.** Docs are emitted set-by-set, so a limited run
> usually ends mid-set; that partial set's `all`/`all_star` (and the list/binary means) are
> computed over fewer questions and are not comparable to a full run. Use `--limit` only for
> plumbing smoke tests; report numbers only from a full-corpus run.

## Deviations from the paper

1. **In-harness embedder.** BeliefQ[Dist.] uses `sentence-transformers/all-roberta-large-v1`
   loaded inside the eval process (adds `sentence-transformers` as a runtime dep; slower).
   This is faithful to the metric, just relocated into the harness.
2. **Sampled decoding → non-deterministic scores.** `do_sample=True, max_new_tokens=365`
   mirrors FANToM's HuggingFace agent (`agents/huggingface.py`). The agent does **not** set
   a temperature (it inherits each model's `generation_config`); lm-eval requires a positive
   temperature under `do_sample`, so we force `temperature=1.0`. For any model whose config
   ships a non-1.0 default this is a (small) deviation. The paper's headline numbers were
   produced with API models under different decoding; run-to-run variance is expected on HF
   models.
3. **`fantom_full` uses `full_context`.** The stored `missed_info_accessibility` labels
   on answerability/info-access QAs are re-derived at flatten time (the raw labels were
   computed for the short excerpt): a list QA is forced `inaccessible` if `wrong_answer`
   is non-empty; binary QAs for a set are all stamped `inaccessible` if any character
   answers "no". Belief and fact QAs are not touched. Scores are expected to be lower
   than `fantom` (harder: more context, more noise).
4. **`fantom_cot` is a single-pass CoT approximation.** FANToM's original protocol is
   two model calls: (1) "Let's think step by step." → reasoning; (2) reasoning +
   "Therefore, the answer is:" → answer. lm-eval has no two-pass mechanism. Instead,
   both cues are baked into the prompt in one shot and the answer is parsed from after
   "Therefore, the answer is:" in the response (falling back to the full response if the
   string is absent). This is the same pattern as `gpqa_cot_zeroshot` in the harness.
4. **Headline scalars only.** The error-analysis diagnostics (`wrong_reasons` frequency
   dicts) and the character-tracking analyses (`set:ALL_character`,
   `character_answer_consistency`) from `score_and_analyze` are not exposed as lm-eval
   metrics (they are dict-valued / diagnostic, not headline numbers).
5. **Control task as metrics, not a second run.** The original dumps a separate
   `control_task` report; here it is the `control_*` metric columns from the same run
   (equally faithful, more convenient).

## Faithfulness validation

Validated against the **actual original** `eval_fantom` code (imported with its unused
heavy/SDK deps stubbed; no model/embedder load), under the same pandas:

- **Prompt assembly:** `utils._flatten` matches the real `setup_fantom` on all 12,832 docs —
  byte-exact `input_text` *and* identical MC `choices_text` + gold index (so the seed-99 RNG
  consumption order matches too).
- **Scoring:** `utils._score_report` matches the real `run_reports`/`score_and_analyze` on
  all 23 headline numbers (set-conjunctions, accessible-set filtering, binary weighted-F1,
  the `no:long` short-input drop).
