## ToMATO

This adapter mirrors the benchmark's local multiple-choice protocol and exposes the
same reporting slices that the original summary script emits.

`tomato_all` is the full released corpus. The top-level `tomato` group runs the full
task plus the order, mental-state, and false-belief slices in one lm-eval command.

Faithfulness notes:
- The prompt wording and answer-label space mirror `benchmarks/ToMATO/code/run_local_llm.py`:
  a system instruction, a `# Transcript / # Question / # Options` user turn, and
  `[A]`-`[D]` continuation scoring.
- The released dataset ships `conversation`, not `transcript`; the adapter uses
  `conversation` as the benchmark's transcript field when `transcript` is absent.
- The top-level `tomato` group is intentionally non-aggregating because its subtasks
  overlap heavily; compare the leaf accuracies directly to the original summary JSON.

Run:

```bash
lm-eval run --model hf   --model_args pretrained=Qwen/Qwen3-8B,dtype=bfloat16,enable_thinking=false   --apply_chat_template   --tasks tomato   --output_path "outputs/ToMATO results/Adaptation"
```
