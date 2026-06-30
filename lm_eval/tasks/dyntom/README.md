# DynToM — lm-eval adapter

Faithful adaptation of the DynToM benchmark (arXiv 2505.17663) for `lm-evaluation-harness`.

## Tasks

| Task | Prompt style | `max_gen_toks` |
|------|-------------|----------------|
| `dyntom` | Vanilla (Fig 17) | 2048 |
| `dyntom_cot` | Chain-of-Thought (Fig 17) | 4096 |

## Architecture

- **doc = one trial (story).** All questions for a trial are batched into a single prompt.
- **Output:** the model returns a JSON dict `{question_id: letter}`.
- **Scoring:** `process_results` parses the JSON and emits 11 per-trial metrics, aggregated by `mean`.
- **Data source:** HF dataset `YangXiao-nlp/DynToM` (`DynToM.json`), fetched via `hf_hub_download`.

## Metrics (11)

| Metric | Questions/trial | Definition |
|--------|----------------|------------|
| `acc` | 71 | **Faithful headline AVG** — all questions incl. type_c. Reproduces paper Table-3 AVG. |
| `acc_core` | 56 | **Ours (not paper):** type_a + type_d only, sans influence. |
| `acc_influence` | 15 | type_c questions only. |
| `acc_belief_u` | 5 | type_a, subject = belief. |
| `acc_belief_t` | 9 | type_d, subject = belief. |
| `acc_emotion_u` | 5 | type_a, subject = emotion. |
| `acc_emotion_t` | 9 | type_d, subject = emotion. |
| `acc_intention_u` | 5 | type_a, subject = intention. |
| `acc_intention_t` | 9 | type_d, subject = intention. |
| `acc_action_u` | 5 | type_a, subject = action. |
| `acc_action_t` | 9 | type_d, subject = action. |

Sanity identity: `acc = (56 * acc_core + 15 * acc_influence) / 71`.

## Running

```bash
# Vanilla
lm-eval run --model hf --model_args pretrained=<model> \
    --tasks dyntom --limit 5 --log_samples --output_path out/

# CoT
lm-eval run --model hf --model_args pretrained=<model> \
    --tasks dyntom_cot --limit 5 --log_samples --output_path out/
```

**Context window requirement:** measured across all 1,111 trials, prompts range from ~10k to ~23k tokens (min 41k chars, max 91k chars at ~4 chars/token). The model must support at least a **24k-token context**; 32k+ is recommended. Smaller models will silently left-truncate the prompt, dropping early questions and tanking scores in a way that resembles a model failure rather than a config issue.

Recommended model for the smoke test: `Qwen/Qwen3-1.7B` (32k context, ~1.7B params):

```bash
lm-eval run --model hf --model_args pretrained=Qwen/Qwen3-1.7B \
    --tasks dyntom --limit 5 --log_samples --output_path out/
```

## Deviations from the paper

1. **Trial set is 1,111 trials (78,881 Q), not 1,100 / 78,100.** The paper filtered by an unrecoverable quality step; we filter by `scenario numbers == 5` directly from the HF release (~1% gap).
2. **Reconstructed prompt serialization.** The paper's Fig 17 prompt wording is reproduced verbatim; however, the exact stringification of `{story}` and `{questions_new}` is a reconstruction (eval code was not released). This adapter uses `json.dumps(..., indent=2)` for both slots.
3. **Reconstructed JSON extraction.** Lenient `json.loads` + `type_X_*_N: letter` regex fallback. Missing or unparseable answers are scored wrong (faithful to no-answer behavior).
4. **Sampled decoding → non-deterministic `acc`** (temperature=0.7, top_p=0.9). The paper reports a single run; run-to-run variance is expected.
5. **`acc_core` is not a paper metric.** It excludes type_c (influence) questions. The faithful reproduction number is `acc`.
6. CoT over 71 answers is long; small models may truncate before the closing `}`, causing those answers to score wrong. This matches the paper's observed score collapse on small models.
