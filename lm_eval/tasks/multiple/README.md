# MultiPL-E

MultiPL-E translates HumanEval and MBPP into multiple programming languages.
This integration initially supports the six languages used by the OLMES task
inventory: C++, Java, JavaScript, PHP, Rust, and shell.

The task definitions and dataset are pinned to:

- Source: `nuprl/MultiPL-E` commit
  `3025a531af7450e7df8b96fe0440e9804480bbad`
- Dataset: `nuprl/MultiPL-E` revision
  `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`

The base task names use the official pass@1 profile: 20 samples at temperature
0.2 and top-p 0.95. Tasks ending in `_pass_at_10_100` use the official
pass@1/pass@10/pass@100 profile: 200 samples at temperature 0.8 and top-p 0.95. The
`multiple_pass_at_1`, `multiple_pass_at_10_100`, and `multiple` groups select
the corresponding profiles. MultiPL-E's prompts, per-language stop strings,
program assembly, and unbiased pass@k estimator are preserved. lm-eval's
`max_gen_toks` is a new-token limit, whereas the reference AutoModel runner's
1,024-token limit includes the prompt; this is the one generation-limit
difference.

## Safe execution

Generated programs are untrusted. Every task is marked `unsafe_code`, so the
lm-eval run must explicitly include `--confirm_run_unsafe_code`. Scoring also
requires `LM_EVAL_MULTIPLE_EXECUTOR` to name an external sandbox executor. The
harness never falls back to running a language toolchain in its own process.

The executor command reads one JSON object per line from standard input. Each
object contains `id`, `name`, `language`, and the complete `program`. It must
write one JSON object per line, in order, with the matching `id` and either:

```json
{"id": 0, "passed": true}
```

or the official MultiPL-E result fields:

```json
{"id": 0, "status": "OK", "exit_code": 0}
```

The command should run the pinned official evaluator in a locked-down
container or isolated worker with no network, a read-only root filesystem,
non-root credentials, per-process CPU/memory/output/time limits, and no host or
Kubernetes credentials. `LM_EVAL_MULTIPLE_EXECUTOR_TIMEOUT` controls the
batch-client timeout and defaults to 600 seconds; it is not a substitute for
per-program limits inside the sandbox.

The upstream container documented by MultiPL-E is
`ghcr.io/nuprl/multipl-e-evaluation`. Pin it by digest in production rather
than using a mutable tag.

## BPB

BPB is intentionally unsupported for these translated tasks. The pinned
dataset exposes `prompt`, `tests`, and stop metadata, but no canonical
translated solution/completion. Its `tests` field is only the execution
reference used by pass@k and must not be scored as a gold continuation. A
future BPB variant would need a separately sourced, versioned corpus of
canonical solutions rather than treating arbitrary passing programs as one
reference.

## Licensing and attribution

The Hugging Face dataset card declares the translated dataset MIT-licensed.
The pinned source repository instead contains a BSD 3-Clause license with an
additional restriction forbidding use of repository contents as model training
data. These statements are not equivalent. This integration copies neither
the translated data nor evaluator implementation into lm-eval; it downloads
the pinned dataset and delegates execution to an operator-provided official
evaluator. Users should review the upstream terms before redistribution or
training use.

Paper: *MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code
Generation* (Cassano et al., 2022), https://arxiv.org/abs/2208.08227
