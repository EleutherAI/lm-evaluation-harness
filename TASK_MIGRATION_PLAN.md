# Qwen3.5 OLMES-BPB evaluation plan

Audit date: 2026-08-20

Baselines:

- `EleutherAI/lm-evaluation-harness@e0ec2b4bfa40aadc42ce3ccf6bb14882071a1a94`
- `allenai/olmes@5a51f502d463b8cdc4a2dcad7d7096c41ff1197e`

The executable campaign definition is
`scripts/olmes_bpb_eval/suite.json`. It is authoritative when this document and
the code differ.

## Scope and completion criteria

Evaluate these 22 requested families plus OLMES Basic Skills on the pinned
`Qwen/Qwen3.5-0.8B-Base` and `Qwen/Qwen3.5-2B-Base` revisions:

| Requested family | lm-eval selector |
|---|---|
| Basic Skills | `basic_skills` |
| ARC Challenge | `arc_challenge` |
| ARC Easy | `arc_easy` |
| BoolQ | `boolq` |
| CSQA | `commonsense_qa` |
| HellaSwag | `hellaswag` |
| PIQA | `piqa` |
| SocialIQA | `social_iqa` |
| WinoGrande | `winogrande` |
| MMLU | `mmlu` |
| MMLU-Pro | `mmlu_pro` |
| GSM8K | `gsm8k` |
| MATH | `minerva_math` |
| HumanEval | `humaneval` |
| MBPP | `mbpp` |
| EvalPlus HumanEval | `humaneval_plus` |
| EvalPlus MBPP | `mbpp_plus` |
| BBH | `bbh` |
| GPQA | `gpqa_main_zeroshot` (the full 448-question split) |
| CRUXEval-O | `cruxeval_output` |
| MultiPL-E | `multiple_pass_at_1` |
| BBEH | `bbeh` |
| SuperGPQA | `supergpqa` |

MRCR and RULER are explicitly excluded. Completion requires:

1. each task's source-native metric without changed prompts or grading;
2. OLMES macro BPB and pooled corpus BPB over the exact canonical target;
3. immutable model, tokenizer, dataset, source, and runtime pins;
4. local validation, scorer parity tests, and a two-example B200 smoke run;
5. full results in a native Google Sheet with leaf metrics and provenance;
6. a reviewable PR.

## Source-backed additions

- Basic Skills: 12 tasks (six RC and six MC) from
  `allenai/basic-skills@faf63e9719e124d7741519a024719c8992622630`,
  including exact deterministic OLMES shuffling, stateless five-shot sampling,
  and true continuation-token `acc_per_token`.
- BBEH: 23 full tasks plus the official 460-example mini file from
  `google-deepmind/bbeh@80d12ca916b7158f22293fcf3144f4d3d854d4be`.
  Report leaf accuracy, micro accuracy, and the paper's adjusted harmonic mean.
- SuperGPQA: official zero/fixed-five-shot prompts and hierarchical metrics from
  `SuperGPQA/SuperGPQA@e31e8bf3e7089aef808be5ddecafd4538dee6e41`
  with dataset revision `4430d4458112c7d4497fdcf94d7cc223313d6acf`.
- MultiPL-E: HumanEval and MBPP in C++, Java, JavaScript, PHP, Rust, and shell
  from source commit `3025a531af7450e7df8b96fe0440e9804480bbad`
  and dataset revision `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`.
  Original pass@k scoring is double-gated behind an external sandbox.

MultiPL-E BPB is N/A: the pinned official release contains prompts and test
harnesses but no canonical translated reference completions. Scoring the tests
as a continuation would produce a number, but it would not be BPB of a source
answer and is deliberately rejected.

## BPB definitions

For prompt `x`, canonical continuation `y`, and natural-log likelihood `L`:

```text
example_bpb = -L / (len(y.encode("utf-8")) * ln(2))
bpb_macro   = mean(example_bpb)
bpb_corpus  = -sum(L) / (sum(len(y.encode("utf-8"))) * ln(2))
```

`bpb_macro` is OLMES benchmark `bits_per_byte_corr`: every example has equal
weight. `bpb_corpus` is the standard pooled-corpus definition used by OLMES's
separate perplexity path. They differ whenever target byte lengths differ.

Multiple-choice tasks reuse the likelihood of the correct option. Generative
tasks add one teacher-forced gold request while preserving the generation used
by the original metric. Canonical code solutions, never test suites, are used
for HumanEval, MBPP, EvalPlus, and CRUXEval. Exact UTF-8 byte counts and exact
backend continuation-token counts are retained as request metadata.

## What is pinned

Pinning is enforcement, not just a note in a README:

- model and tokenizer revisions are passed to the backend;
- dataset revisions are passed in task `dataset_kwargs`;
- source URLs contain immutable Git commits;
- `scripts/olmes_bpb_eval/constraints.txt` requires exact tested package
  versions;
- each result records the lm-eval commit, dirty state, complete source snapshot
  hash, task config, seeds, backend settings, and installed versions.

An upgrade is a new campaign manifest and smoke qualification, not an in-place
install against mutable `main` branches.

## Execution sequence

1. Run task validation, focused pytest, registry tests, ruff, and diff checks.
2. Run a two-example `arc_easy` smoke shard on both pinned models.
3. Run the non-code phase on 16 B200s: eight one-GPU replicas per model.
4. Run Python and MultiPL-E functional metrics only in isolated CPU sandbox
   jobs with no network, no service-account token, non-root identity, a
   read-only root filesystem, and strict process/memory/time limits.
5. Merge shard JSONs into family summary, leaf-task metrics, and a run manifest.
6. Import the workbook as a native Google Sheet and verify the written ranges.
7. Prepare logical commits and open a draft PR for review.

The 0.8B and 2B checkpoints each fit on one B200 in BF16. Sixteen GPUs are a
throughput allocation: eight replicas per checkpoint, not a memory requirement.
One B200 is sufficient for either individual model; code compilation/execution
is CPU-bound and isolated from model inference.

## Known operational gates

- GPQA is gated on Hugging Face and requires the mounted cluster token.
- Existing HumanEval/MBPP/EvalPlus/CRUXEval evaluators execute generated Python;
  they must not run in the model worker process.
- MultiPL-E requires `LM_EVAL_MULTIPLE_EXECUTOR` pointing to the external JSONL
  sandbox adapter documented in `lm_eval/tasks/multiple/README.md`.
- The B200 training-campaign skill requires online W&B loss curves and its
  whole-repository literal-credential scanner flags lm-eval's generic API-token
  parameter code. That workflow does not match an evaluation-only Ray job; do
  not bypass its guard silently. Use an explicitly approved evaluation launch
  path or a scanner-compatible source snapshot.
