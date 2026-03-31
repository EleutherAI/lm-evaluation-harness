# Installation

## Basic install

```bash
pip install lm-eval
```

The base package includes the evaluation harness and CLI. To use specific model backends, install the corresponding extras.

## Extras

| Extra | What it adds | Install command |
|---|---|---|
| `hf` | HuggingFace Transformers models | `pip install lm-eval[hf]` |
| `vllm` | vLLM inference engine | `pip install lm-eval[vllm]` |
| `api` | API-based models (OpenAI, Anthropic, etc.) | `pip install lm-eval[api]` |
| `all` | All model backends | `pip install lm-eval[all]` |
| `dev` | Development dependencies (testing, linting) | `pip install lm-eval[dev]` |

You can combine extras:

```bash
pip install lm-eval[hf,vllm]
```

## Development install

To work on the harness source code or contribute tasks:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[dev]"
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `LMEVAL_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `WARNING` |
| `LM_HARNESS_CACHE_PATH` | Directory for cached model responses | `lm_eval/caching/.cache` |
| `LM_EVAL_DATASET_DIR` | Local fallback directory for datasets — if set, the harness checks here for local copies before downloading from HuggingFace Hub | Not set |
| `HF_TOKEN` | HuggingFace Hub token for gated datasets/models | Not set |

!!! tip
    For debugging task configurations, set `export LMEVAL_LOG_LEVEL="DEBUG"` before running evaluations. This shows prompt rendering, dataset loading, and scoring details.

## Verify installation

```bash
# Check the CLI works
lm-eval --help

# List available tasks
lm-eval ls tasks

# Validate a specific task config
lm-eval validate --tasks hellaswag
```
