# OLMES BPB evaluation campaign

This directory defines the reproducible Qwen3.5 evaluation campaign requested
for the 22 benchmark families in `suite.json`, plus the separately requested
OLMES Basic Skills suite. MRCR and RULER are deliberately excluded.

## What pinning means

Recording a version is provenance; enforcing a version is pinning. This
campaign does both:

- `constraints.txt` makes the installer require exact runtime versions.
- `suite.json` passes immutable model revisions and records dataset/source
  revisions. The selected built-in task YAMLs also pass those dataset revisions
  to Hugging Face rather than loading a mutable branch.
- Every result shard records the lm-eval Git commit, dirty-state source hash,
  model revision, tokenizer revision, package versions, prompts, seeds, and task
  configs.

Pinning prevents a future install or mutable Hub `main` branch from silently
changing scores. It does not promise compatibility forever: upgrades should be
made deliberately, validated on a smoke slice, and recorded as a new campaign
manifest.

## Metrics

`--compute-bpb` keeps every task's original metric and adds:

- `bpb_macro`: the OLMES mean of per-example negative log-likelihood divided by
  UTF-8 bytes and `ln(2)`;
- `bits_per_byte_corr`: the OLMES-compatible name for the same benchmark value;
- `bpb_corpus`: pooled negative log-likelihood divided by pooled UTF-8 bytes and
  `ln(2)`.

MultiPL-E is the principled exception. Its official release has prompts and
tests but no canonical translated reference completions, so BPB is undefined
and reported as N/A. Its tests are never substituted as a fake BPB target.

For both Basic Skills prompt forms (RC and MC), the pinned OLMES task
registration uses `acc_per_token` as the original headline metric. Raw `acc`
remains available as a diagnostic metric.

## Ray launch

The launch is non-interactive and uses one GPU per shard. A smoke run should be
completed before a full phase:

```bash
scripts/olmes_bpb_eval/run_ray.sh smoke /mnt/shared/sravya/olmes-bpb/smoke --limit 2
scripts/olmes_bpb_eval/run_ray.sh non_code /mnt/shared/sravya/olmes-bpb/full
```

The two pinned models run concurrently with eight replicas each by default,
using all 16 booked B200 GPUs. The models need only one B200 each; the other GPUs
are for throughput, not memory. `ray_driver.py` resolves group/tag selectors to
leaf tasks and balances expensive generation tasks across persistent shards.

Gated families can be split from the public-data run without changing the
release suite or colliding with its result files:

```bash
scripts/olmes_bpb_eval/run_ray.sh non_code results \
  --exclude-family gpqa --shard-prefix non-code-public
scripts/olmes_bpb_eval/run_ray.sh non_code results \
  --include-family gpqa --shard-prefix gpqa
```

The `python_code` and `multiple` phases are double-gated and must only be run
with their external sandbox configured. Generated programs must not execute in
the Ray model-serving process. Python tasks require `LM_EVAL_PYTHON_EXECUTOR`
using the protocol in `lm_eval/tasks/CODE_EXECUTION.md`. MultiPL-E additionally
requires the JSONL adapter documented in `lm_eval/tasks/multiple/README.md`.

After fetching the shard directory, build the spreadsheet with:

```bash
python scripts/olmes_bpb_eval/merge_results.py \
  --results-dir results \
  --output qwen35-olmes-bpb.xlsx
```

The workbook contains family summaries, every original/BPB leaf-task metric,
and the complete pin/runtime manifest. The exporter computes the official BBEH
adjusted harmonic mean from the 23 per-task accuracies and exact family corpus
BPB from pooled log-likelihood and byte totals.
