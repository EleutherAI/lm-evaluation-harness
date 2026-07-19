# Helium benchmarks (lm-evaluation-harness)

Tasks for [Helium Trades](https://heliumtrades.com/benchmarks/) open benchmarks on Hugging Face.

## Tasks

| Task | Items | Description |
|------|-------|-------------|
| `helium_market_resolution` | 300 | Option-chain math (IV, delta, MCQ). Partial-credit IV scoring. |
| `helium_market_resolution_mini` | 20 | Smoke-test subset. |
| `helium_model_worldview` | 304 | Cue-swap / worldview prompts (Likert, free-text, essays). |
| `helium_model_worldview_behavioral` | 18 | Refusal asymmetry, sycophancy, both-sidesism only. |
| `helium_model_worldview_mini` | 20 | Smoke-test subset. |

Run the group:

```bash
lm_eval --model hf --model_args pretrained=gpt2 --tasks helium --limit 5
```

Full benchmark:

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks helium_market_resolution --batch_size auto
```

## Scoring notes

**Market Resolution** scoring mirrors the public [methodology](https://heliumtrades.com/benchmarks/): MCQ exact match, IV/delta linear decay by regime tolerance. `core_score` averages core-tier items only.

**Model Worldview** per-item scores are module-dependent (Likert rubric normalization, refusal/sycophancy pattern detection). Cue-swap **flip rate** across `pair_id` groups requires post-processing; not computed inline.

## Datasets

- https://huggingface.co/datasets/HeliumTrades/helium-market-resolution-benchmark
- https://huggingface.co/datasets/HeliumTrades/helium-model-worldview-benchmark

Maintained by the Helium Trades team. PRs welcome.
