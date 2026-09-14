# TemplateAPI Usage Guide

The `TemplateAPI` class is a versatile superclass designed to facilitate the integration of various API-based language models into the lm-evaluation-harness framework. This guide will explain how to use and extend the `TemplateAPI` class to implement your own API models. If your API implements the OpenAI API you can use the `local-completions` or the `local-chat-completions` (defined [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py)) model types, which can also serve as examples of how to effectively subclass this template.

## Overview

The `TemplateAPI` class provides a template for creating API-based model implementations. It handles common functionalities such as:

- Tokenization (optional)
- Batch processing
- Caching
- Retrying failed requests
- Parsing API responses

To use this class, you typically need to subclass it and implement specific methods for your API.

## Key Methods to Implement

When subclassing `TemplateAPI`, you need to implement the following methods:

1. `_create_payload`: Creates the JSON payload for API requests.
2. `parse_logprobs`: Parses log probabilities from API responses.
3. `parse_generations`: Parses generated text from API responses.

Optional Properties:

4. `header`: Returns the headers for the API request.
5. `api_key`: Returns the API key for authentication (if required).

You may also need to override other methods or properties depending on your API's specific requirements.

> [!NOTE]
> Currently loglikelihood and MCQ based tasks (such as MMLU) are only supported for completion endpoints. Not for chat-completion — those that expect a list of dicts — endpoints! Completion APIs which support instruct tuned models can be evaluated with the `--apply_chat_template` option in order to simultaneously evaluate models using a chat template format while still being able to access the model logits needed for loglikelihood-based tasks.

## Evaluate a hosted chat-completions server

Use `local-chat-completions` for a server that accepts OpenAI-compatible chat requests. Despite the adapter's name, the server can be remote. Set `base_url` to the **full** `/v1/chat/completions` URL: the adapter posts to that URL unchanged. Use `OPENAI_API_KEY` for the server's bearer token and `--apply_chat_template` to send structured messages. With `tokenizer_backend=None`, the server applies its own model template and tokenizer.

Choose a `generate_until` task such as `gsm8k_cot_zeroshot`. Chat completions cannot evaluate `loglikelihood` or `multiple_choice` tasks. For those tasks, use `local-completions` with a matching tokenizer and a completion server that returns echoed **prompt** logprobs; generated-token logprobs alone are insufficient.

### Example: vLLM on a Nebius Serverless Endpoint

First deploy the model using the [vLLM Nebius Serverless guide](https://docs.vllm.ai/en/latest/deployment/frameworks/nebius/). It covers the pinned serving image, managed HTTPS URL, authentication, readiness and cleanup. Use its `Qwen/Qwen3-0.6B` model with both model and tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca` and a 4096-token context. The client below runs on a CPU machine; it does not create cloud resources or need the harness's `[vllm]` extra.

Before evaluating, require a successful authenticated `/health` response, `/v1/models` listing the expected model, and a short chat generation. Follow the deployment guide's missing/wrong-token checks. The Endpoint's `RUNNING` state alone does not establish that the model has loaded.

From a harness source checkout in an isolated Python environment, install the API dependencies:

```bash
python -m pip install -e '.[api]'
```

Load `ENDPOINT_URL` with the discovered HTTPS root and `ENDPOINT_TOKEN` with its private token. The token is separate from Nebius CLI/IAM credentials; `OPENAI_API_KEY` is the adapter's environment-variable name. Keep secrets out of `--model_args`, which is logged and saved with results, and disable shell tracing.

This example evaluates the first eight test documents with zero few-shot examples and at most 512 output tokens per document. It is an integration smoke test: **do not report its score as a GSM8K benchmark**. The task's prompts, target and answer filters are preserved; only the dataset revision and local task name are overridden.

```bash
set -euo pipefail
: "${ENDPOINT_URL:?Set the Endpoint HTTPS root URL}"
: "${ENDPOINT_TOKEN:?Load the Endpoint token privately}"
export OPENAI_API_KEY="$ENDPOINT_TOKEN"
umask 077
export RUN_DIR="$(mktemp -d ./lm-eval-nebius.XXXXXX)"

python - <<'PY'
import os
from pathlib import Path

import yaml

source = Path("lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml")
config = yaml.safe_load(source.read_text())
config["task"] = "gsm8k_nebius_smoke"
config["dataset_kwargs"] = {
    "revision": "740312add88f781978c0658806c59bc2815b9866"
}
Path(os.environ["RUN_DIR"], "task.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY

lm-eval run --model local-chat-completions \
    --model_args "model=Qwen/Qwen3-0.6B,base_url=${ENDPOINT_URL%/}/v1/chat/completions,tokenizer_backend=None,tokenized_requests=False,num_concurrent=1,max_retries=1,timeout=120,seed=1234" \
    --tasks "$RUN_DIR/task.yaml" \
    --limit 8 --num_fewshot 0 --batch_size 1 --apply_chat_template \
    --gen_kwargs '{"max_gen_toks":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    --seed 0,1234,1234,1234 --log_samples \
    --output_path "$RUN_DIR/results"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_DIR"], "results")
aggregates = list(root.rglob("results_*.json"))
samples = list(root.rglob("samples_*.jsonl"))
assert len(aggregates) == len(samples) == 1, "Missing or ambiguous outputs"
result = json.loads(aggregates[0].read_text())
assert "gsm8k_nebius_smoke" in result["results"], "Wrong task"
rows = [json.loads(line) for line in samples[0].read_text().splitlines()]
assert len(rows) == 16, "Expected eight documents times two answer filters"
for name in ("strict-match", "flexible-extract"):
    filtered = [row for row in rows if row["filter"] == name]
    assert len(filtered) == 8
    assert {row["doc_id"] for row in filtered} == set(range(8))
assert all(
    isinstance(row["resps"][0][0], str) and row["resps"][0][0].strip()
    for row in rows
), "Empty or invalid generation"
print(f"Verified eight documents and both answer filters in {root}")
PY
```

`enable_thinking=false` is specific to this Qwen model. Inspect the raw responses when changing the model or generation limit: reasoning-only or null content can become empty output, and a truncated answer is not a reliable evaluation. Also inspect both `resps` and `filtered_resps`: arbitrary prose or LaTeX can fail the task's numeric answer filters even when the raw response is nonempty. The task supplies stop strings including `<|im_end|>`. The chat adapter does not count input tokens or enforce a context limit when tokenization is disabled. Check the rendered prompts plus the output budget against the server's context size; setting `max_length` on this client does not perform that check.

`max_retries=1` means **one total attempt**, disabling automatic retries. Larger values retry permanent HTTP errors such as 401 as well as transient failures; an ambiguous timeout can repeat inference. `timeout=120` limits individual network waits, not total evaluation time or Endpoint lifetime. Inspect failures before deliberately rerunning into a fresh output directory.

The result writer can log a save failure without raising it, so verify the JSON and JSONL files even if the command exits successfully. There are 16 sample rows because each of eight documents is logged for two answer filters. Keep both files and the generated task YAML. Record the harness commit, dependency versions, image digest, model/tokenizer revisions, server template/settings and dataset revision with the run. The two seed options set harness RNGs and the API request seed respectively; they do not guarantee identical GPU outputs, even for repeated requests to the same server. This example uses no request/response cache, so an interrupted run may require repeating inference.

After the run, use the deployment guide to stop or delete the Endpoint you own and confirm the operation completes. Ending the harness process does not stop the Endpoint or its billing. Keep results on the client or explicitly export and verify them in durable storage; the serving container's disk is not an evaluation-results store.

## TemplateAPI Arguments

When initializing a `TemplateAPI` instance or a subclass, you can provide several arguments to customize its behavior. Here's a detailed explanation of some important arguments:

- `model` or `pretrained` (str):
  - The name or identifier of the model to use.
  - `model` takes precedence over `pretrained` when both are provided.

- `base_url` (str):
  - The base URL for the API endpoint.

- `tokenizer` (str, optional):
  - The name or path of the tokenizer to use.
  - If not provided, it defaults to using the same tokenizer name as the model.

- `num_concurrent` (int):
  - Number of concurrent requests to make to the API.
  - Useful for APIs that support parallel processing.
  - Default is 1 (sequential processing).

- `timeout` (int, optional):
  - Timeout for API requests in seconds.
  - Default is 300.

- `tokenized_requests` (bool):
  - Determines whether the input is pre-tokenized. Defaults to `True`.
  - Requests can be sent in either tokenized form (`list[list[int]]`) or as text (`list[str]`, or `str` for batch_size=1).
  - For loglikelihood-based tasks, prompts require tokenization to calculate the context length. If `False` prompts are decoded back to text before being sent to the API.
  - Not as important for `generate_until` tasks.
  - Ignored for chat formatted inputs (list[dict...]) or if tokenizer_backend is None.

- `tokenizer_backend` (str, optional):
  - Required for loglikelihood-based or MCQ tasks.
  - Specifies the tokenizer library to use. Options are "tiktoken", "huggingface", or None.
  - Default is "huggingface".

- `max_length` (int, optional):
  - Maximum length of input + output.
  - Default is 2048.

- `max_retries` (int, optional):
  - Maximum total number of attempts for failed API requests, including the initial attempt.
  - Default is 3. Set to 1 to disable automatic retries. HTTP errors such as 401 are retried too.

- `max_gen_toks` (int, optional):
  - Maximum number of tokens to generate in completion tasks.
  - Default is 256 or set in task yaml.

- `batch_size` (int or str, optional):
  - Number of requests to batch together (if the API supports batching).
  - Can be an integer or "auto" (which defaults to 1 for API models).
  - Default is 1.

- `seed` (int, optional):
  - Random seed for reproducibility.
  - Default is 1234.

- `add_bos_token` (bool, optional):
  - Whether to add the beginning-of-sequence token to inputs (when tokenizing).
  - Default is False.

- `custom_prefix_token_id` (int, optional):
  - Custom token ID to use as a prefix for inputs.
  - If not provided, uses the model's default BOS or EOS token (if `add_bos_token` is True).

- `verify_certificate` (bool, optional):
  - Whether to validate the certificate of the API endpoint (if HTTPS).
  - Default is True.

- `header` (dict, optional):
  - Custom headers for API requests.
  - If not provided, uses `{"Authorization": f"Bearer {self.api_key}"}` by default.

Example usage:

```python
class MyAPIModel(TemplateAPI):
    def __init__(self, **kwargs):
        super().__init__(
            model="my-model",
            base_url="https://api.mymodel.com/v1/completions",
            tokenizer_backend="huggingface",
            num_concurrent=5,
            max_retries=5,
            batch_size=10,
            **kwargs
        )

    # Implement other required methods...
```

When subclassing `TemplateAPI`, you can override these arguments in your `__init__` method to set default values specific to your API. You can also add additional (potentially user-specified) arguments as needed for your specific implementation.

## Example Implementation: OpenAI API

The `OpenAICompletionsAPI` and `OpenAIChatCompletion` ([here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py) classes demonstrate how to implement API models using the `TemplateAPI` class. Here's a breakdown of the key components:

### 1. Subclassing and Initialization

```python
@register_model("openai-completions")
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )
```

### 2. Implementing API Key Retrieval

```python
@cached_property
def api_key(self):
    key = os.environ.get("OPENAI_API_KEY", None)
    if key is None:
        raise ValueError(
            "API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    return key
```

### 3. Creating the Payload

```python
def _create_payload(
    self,
    messages: Union[List[List[int]], List[dict], List[str], str],
    generate=False,
    gen_kwargs: Optional[dict] = None,
    **kwargs,
) -> dict:
    if generate:
        # ... (implementation for generation)
    else:
        # ... (implementation for log likelihood)
```

### 4. Parsing API Responses

```python
@staticmethod
def parse_logprobs(
    outputs: Union[Dict, List[Dict]],
    tokens: List[List[int]] = None,
    ctxlens: List[int] = None,
    **kwargs,
) -> List[Tuple[float, bool]]:
    # ... (implementation)

@staticmethod
def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
    # ... (implementation)
```

The requests are initiated in the `model_call` or the `amodel_call` methods.

## Implementing Your Own API Model

To implement your own API model:

1. Subclass `TemplateAPI` or one of its subclasses (e.g., `LocalCompletionsAPI`).
2. Override the `__init__` method if you need to set specific parameters.
3. Implement the `_create_payload` and `header` methods to create the appropriate payload for your API.
4. Implement the `parse_logprobs` and `parse_generations` methods to parse your API's responses.
5. Override the `api_key` property if your API requires authentication.
6. Override any other methods as necessary to match your API's behavior.

## Best Practices

1. Use the `@register_model` decorator to register your model with the framework (and import it in `lm_eval/models/__init__.py`!).
2. Use environment variables for sensitive information like API keys.
3. Properly handle batching and concurrent requests if supported by your API.
