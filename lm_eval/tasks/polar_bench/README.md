# POLAR-Bench

[Paper](https://arxiv.org/abs/2605.19127) |
[Dataset](https://huggingface.co/datasets/Qiaoyuan/POLAR-Bench) |
[Reference code](https://github.com/qiaoyuan667/POLAR-Bench)

POLAR evaluates selective disclosure: an evaluated model A follows a privacy policy
while interacting with an adversarial third party. The released dataset contains
7,852 samples across 10 domains, 5 attack protocols, and 5 privacy levels.
Protocols 1–4 replay fixed attacks; protocol 5 calls a fixed external attacker B.
B is not a judge. Scoring uses deterministic matching over all A turns.

This integration supports **non-reasoning instruction models A**. Visible answers
enter subsequent A/B history and scoring. Separate B `reasoning` fields are never
forwarded; B must return its final attack in `message.content`. Unexpected inline
thinking tags are stripped defensively, but reasoning-prefilled A templates are
outside the supported scope. A chat template must retain the entire per-sample
system policy; discarded or transformed policies raise instead of being scored.
A separate rendering-only probe, using a controlled user message, detects dropped
system roles even when an attacker echoes the policy in its own message. The
probe never reaches A inference or either peer's history. These checks do not
certify every model's interpretation of role hierarchy.

## Run

Use an existing GPU allocation and a reachable OpenAI-compatible B chat endpoint.
Install this checkout and the optional dependencies:

```bash
pip install -e '.[vllm,polar_bench]'
```

If a container already supplies compatible vLLM packages, install
`pip install -e '.[polar_bench]'` in an environment that exposes those packages.
Then configure the served B model ID and run a non-reasoning A:

```bash
export POLAR_ATTACKER_BASE_URL='http://YOUR_B_HOST:8080/v1'
export POLAR_ATTACKER_MODEL='meta-llama/Llama-3.3-70B-Instruct'
# Set POLAR_ATTACKER_API_KEY securely if the service requires authentication.
export POLAR_VLLM_ARGS='tensor_parallel_size=4,data_parallel_size=1,max_model_len=32768'
export POLAR_OUTPUT_PATH='results/polar-qwen25-7b'
bash scripts/run_polar_vllm.sh Qwen/Qwen2.5-7B-Instruct
```

The helper neither allocates GPUs nor starts B. Slurm, CSCS, and W&B are not
required. The first full run downloads the pinned JSON from HF revision
`2fe6a18ac50ab86c64b2e5935031901ca35a2d18`; its SHA256 and row count are recorded.
Code is MIT licensed; the dataset is CC BY 4.0. Invalid rows fail explicitly.
For a smoke run, set `POLAR_DATASET_PATH` to a local subset in the same JSON-array
format. Use a subset covering all protocols and domains; a limit is only for
integration checks, not a full benchmark score.

The task is also available through the normal CLI:

```bash
export POLAR_MODEL_A_NAME='Qwen/Qwen2.5-7B-Instruct'
lm-eval run --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,enable_thinking=false,tensor_parallel_size=4,max_model_len=32768 \
  --tasks polar_bench --apply_chat_template --num_fewshot 0 \
  --log_samples --output_path results/polar_bench
```

Use one harness process (`world_size=1`); tensor parallelism within the model is
supported. Data parallel/distributed dialogue evaluation is not implemented.
Do not add a global system instruction or few-shot examples: the task supplies
its own per-sample policy and complete conversation.

## Protocol and reproducibility

- A generates up to 8,192 tokens per turn by default. Override with
  `--gen_kwargs max_gen_toks=16384` after the helper's model argument if needed.
- B's default is 32,768 tokens per request. Override with
  `POLAR_ATTACKER_MAX_TOKENS`; ensure B's context fits both input and requested
  output. This budget includes reasoning for servers that count it toward output.
  Truncated or empty B responses fail explicitly. No finite budget guarantees
  completion.
- B requests are synchronous and sequential within each wave. Full-dataset
  throughput depends on the service; concurrent attacker scheduling is deferred.
- B's timeout defaults to 600 seconds per attempt, with two SDK retries. Override
  with `POLAR_ATTACKER_TIMEOUT`.
- `POLAR_MAX_ROUNDS` defaults to 6, with sample-specific turn budgets taking
  precedence. Fixed scripts are capped by their length. B's exact `<STOP>` ends
  an episode before the next A request. A refusal or leak alone does not stop it.
- `POLAR_SEED` defaults to 42. Record A/B names, weights/revisions, serving
  settings, budgets, and the dataset hash. `POLAR_MODEL_A_NAME` participates in
  per-turn seed derivation; the helper sets it from its model argument.
- The reference helpers in `utils.py` are copied from public evaluator commit
  `5719f1ae35b0a5d4a4832fad2d59b060b41669f5`, with source SHA256 and MIT notice.
  Their ASTs match the published functions. The increased A/B budgets differ
  from the reference's 1,200/500; these results are not direct reproductions of
  the historical configuration. The task records protocol `final-answer-only-v2`.
- Request caching (`--cache_requests`) incorporates dataset SHA256, task
  configuration, and sample selection, preventing stale requests after changing
  local data or generation settings. Clients and episode state are never cached.
- The standard interface returns text without A's termination reason, so this
  task does not claim an A truncation-rate metric. Check token/context limits.
  Response caching (`--use_cache`) is not validated for this integration.

Metrics `privacy`, `utility`, and `overall` range from 0 to 1 (higher is better).
Overall is the mean of privacy and utility. `final_answer_rate` reports the share
of A turns containing visible answers; `zero_turn_rate` reports empty episodes.
An empty episode may score privacy 1, utility 0, overall 0.5, so report both
completion diagnostics alongside the three primary metrics.
