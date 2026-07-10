## PersuasiveToM

This adapter keeps the benchmark's eight released splits as separate lm-eval leaves
under the `persuasivetom` group and mirrors the original non-CoT evaluation prompt.

The adapter uses `generate_until` rather than letter-loglikelihood scoring. The
system instruction is carried by `description`, the user turn matches
`benchmarks/PersuasiveToM/evaluation/evaluate.py`, and `process_results` uses the
same answer-letter extraction pattern as the local original-eval bridge.

Faithfulness notes:
- Malformed `answerKey` rows are kept in the denominator and score incorrect, matching
  the original benchmark behavior on `intent_er`.
- The adapter does not reproduce the benchmark's retry-until-valid loop. Invalid
  generations are scored wrong on the first try instead of being resampled.
- The top-level `persuasivetom` group is still a convenience micro-average across the
  eight leaves; the faithful reporting surface is the per-leaf accuracy table.

Run:

```bash
lm-eval run --model hf   --model_args pretrained=Qwen/Qwen3-8B,dtype=bfloat16,enable_thinking=false   --apply_chat_template   --tasks persuasivetom   --output_path outputs/persuasivetom_results/Adaptation
```
