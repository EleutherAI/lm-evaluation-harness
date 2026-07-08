# NegotiationToM adapter

This adapter loads the password-protected `benchmarks/NegotiationToM/NegotiationToM.zip` archive, reconstructs the paper prompts, and exposes the benchmark as three lm-eval tasks:

- `negotiationtom_belief` - multiple choice over A-D letters
- `negotiationtom_desire` - multiple choice over A-D letters
- `negotiationtom_intention` - `generate_until` with multi-label parsing and micro/macro-F1

The benchmark group `negotiationtom` aggregates the three leaves.

## What is faithful here

- The prompt text is reconstructed from the paper's baseline templates.
- Desire and belief use the paper's A-D answer format and the paper's camping-trip framing.
- Intention uses the paper's A-I strategy inventory, targets the first one or two utterances named by the archive, and accepts either letters or label names in generation output.
- The archive is read with the published password `NegotiationToM`.

## Known deviations

- The paper also reports CoT and few-shot variants; this adapter currently implements the zero-shot baseline only.
- The paper's headline `all-correct` and dialogue-level consistency metrics are not represented as native lm-eval scalars here. The leaf tasks expose per-doc scores and the group reports leaf means.
- Intention label names are canonicalized to the dataset's stored labels during scoring.

## Run

Use the batch runner from the repo root:

```bash
python benchmark_runner.py \
  --tasks negotiationtom \
  --model hf \
  --model_args pretrained=Qwen/Qwen3-1.7B \
  --device cuda:0 \
  --batch_size auto \
  --output_path outputs/negotiationtom
```

For a quick plumbing check, add `--limit 20`.
