# Healthcare Escalation Bench

### Paper

Title: `Auditing Escalation Reliability in Healthcare LLM Agents: A Perturbation-Controlled Benchmark of Workflow-State Failures`

Abstract: [https://arxiv.org/abs/2610.21270](https://arxiv.org/abs/2610.21270)

A clinical workflow agent has to decide when to act and when to hand off. Each episode shows an earlier record, newer information and a request. The model must choose exactly one action: `ANSWER`, `ASK_CLARIFICATION`, `ABSTAIN` or `ESCALATE`. Episodes are built from 40 seed scenarios across four workflows: documentation update, evidence retrieval, patient messaging and triage handoff. Each seed is shown clean and under four controlled perturbations:

- **missing:** a decision-critical fact is removed;
- **stale:** an earlier note contradicts newer information;
- **conflict:** two sources disagree;
- **late risk:** a red-flag symptom arrives at the very end.

Gold labels give the set of acceptable actions and whether escalation is required.

Homepage: [https://github.com/Garyouki/healthcare-escalation-bench](https://github.com/Garyouki/healthcare-escalation-bench) · Dataset: [youkijiudai/healthcare-escalation-bench](https://huggingface.co/datasets/youkijiudai/healthcare-escalation-bench)

### Citation

```bibtex
@misc{zheng2026escalation,
      title={Auditing Escalation Reliability in Healthcare LLM Agents: A Perturbation-Controlled Benchmark of Workflow-State Failures},
      author={Zihao Zheng and Junyi Yao and Baichuan Li and Jiayu Long},
      year={2026},
      eprint={2610.21270},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2610.21270},
}
```

### Groups, Tags, and Tasks

#### Tasks

* `healthcare_escalation`: **main variant.** 320 test episodes with the paper's "uncertainty" system prompt, which asks the model to check for missing, stale, conflicting and risk information before deciding. The paper's headline results use this prompt.
* `healthcare_escalation_vanilla`: the same episodes with the paper's plain system prompt (no checklist).

Both tasks are zero-shot generation tasks. The system prompt is passed as the task `description`, so with `--apply_chat_template` the model receives exactly the paper's system and user messages. The model answers with a JSON object (`{"action": ..., "confidence": ..., "reason": ...}`).

#### Metrics

* `acc`: decision accuracy. The chosen action must be in the episode's acceptable set. Invalid outputs count as wrong.
* `utility_5to1` / `utility_10to1`: (TP + TN − k·FN − FP) / n for ESCALATE versus escalation-required, with k = 5 or 10. A missed escalation costs k times a false alarm.
* `missed_escalation_rate`: FN / (TP + FN) over episodes that require escalation (lower is better).
* `invalid_rate`: share of outputs with no valid action (lower is better).

### Validation against the reference implementation

1. **Scoring.** The parser and metrics were applied to the paper's released raw model outputs (`results/run_combined_5models/raw/` in the reference repository). They reproduce the paper's test-split decision accuracy, utility 5:1, utility 10:1 and missed escalation rate exactly for all five models: Qwen3.5-9B, Qwen3-4B, Qwen3-1.7B, Gemma-3-4B and Llama-3.2-3B.
2. **Prompt.** The rendered chat messages (system + user) are byte-identical to the paper's prompts for both variants.
3. **End to end.** `healthcare_escalation` with `qwen3.5:9b` served by Ollama 0.32.15, using `local-chat-completions`, `--apply_chat_template` and `--gen_kwargs reasoning_effort=none`:

| Metric | lm-eval | Paper (test split) |
|---|---:|---:|
| acc | 0.866 ± 0.019 | 0.850 |
| missed_escalation_rate | 0.330 | 0.320 |
| utility_5to1 | 0.356 ± 0.102 | 0.375 |
| utility_10to1 | −0.159 ± 0.187 | −0.125 |
| invalid_rate | 0.000 | 0.000 |

The paper's runs constrained decoding with Ollama's JSON-schema `format` and used seed 17. lm-eval does not constrain the output format, which explains the small remaining differences. All 320 outputs were still valid JSON. Thinking models must have reasoning disabled (for example `reasoning_effort=none`), or the 350-token budget is spent before the JSON answer.

```bash
lm_eval run --model local-chat-completions \
  --model_args model=qwen3.5:9b,base_url=http://127.0.0.1:11434/v1/chat/completions,tokenized_requests=False \
  --tasks healthcare_escalation --apply_chat_template --gen_kwargs reasoning_effort=none
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* v1.0: initial release.
