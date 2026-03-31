# API Model Backend

The `TemplateAPI` class facilitates integration of API-based language models into the evaluation harness. If your API implements the OpenAI API, you can use the built-in `local-completions` or `local-chat-completions` model types directly. Otherwise, subclass `TemplateAPI` to implement your own.

!!! tip
    For non-API models or when you need lower-level control over inference, see [Custom Model Backend](custom_model.md) instead.

## Overview

`TemplateAPI` handles common functionality:

- Tokenization (optional)
- Batch processing
- Caching
- Retrying failed requests
- Parsing API responses

## Key methods to implement

When subclassing `TemplateAPI`, implement:

1. **`_create_payload`** — Creates the JSON payload for API requests
2. **`parse_logprobs`** — Parses log probabilities from API responses
3. **`parse_generations`** — Parses generated text from API responses

Optional properties:

4. **`header`** — Returns headers for the API request
5. **`api_key`** — Returns the API key for authentication

!!! note
    Loglikelihood and multiple-choice tasks (such as MMLU) are only supported for **completion** endpoints, not for chat-completion endpoints. Completion APIs supporting instruct-tuned models can use `--apply_chat_template` to evaluate with a chat template format while still accessing model logits.

## TemplateAPI arguments

| Argument | Description |
|---|---|
| `model` / `pretrained` | Model name or identifier. `model` takes precedence. |
| `base_url` | Base URL for the API endpoint. |
| `tokenizer` | Tokenizer name/path. Defaults to the model name. |
| `num_concurrent` | Number of concurrent API requests. |
| `max_retries` | Maximum number of retry attempts for failed requests. |
| `timeout` | Request timeout in seconds. |
| `max_gen_toks` | Maximum number of tokens to generate. |
| `batch_size` | Batch size for processing requests. |

## Example: OpenAI-compatible API

For APIs that follow the OpenAI format, use the built-in model types directly:

```bash
# Completion endpoint
lm-eval run \
    --model local-completions \
    --model_args model=my-model,base_url=http://localhost:8000/v1/completions,num_concurrent=10 \
    --tasks hellaswag

# Chat completion endpoint
lm-eval run \
    --model local-chat-completions \
    --model_args model=my-model,base_url=http://localhost:8000/v1/chat/completions \
    --tasks hellaswag \
    --apply_chat_template
```

## Example: Custom API subclass

```python
from lm_eval.models.api_models import TemplateAPI
from lm_eval.api.registry import register_model

@register_model("my_api")
class MyAPIModel(TemplateAPI):

    def _create_payload(self, messages, gen_kwargs, *, seed=None, **kwargs):
        """Build the request payload."""
        return {
            "model": self.model,
            "prompt": messages,
            "max_tokens": gen_kwargs.get("max_gen_toks", 256),
            "temperature": gen_kwargs.get("temperature", 0),
        }

    def parse_logprobs(self, outputs, **kwargs):
        """Extract log probabilities from the API response."""
        return [output["logprobs"]["token_logprobs"] for output in outputs]

    def parse_generations(self, outputs, **kwargs):
        """Extract generated text from the API response."""
        return [output["choices"][0]["text"] for output in outputs]

    @property
    def header(self):
        return {"Authorization": f"Bearer {self.api_key}"}
```

## Reference implementations

- `lm_eval/models/openai_completions.py` — OpenAI completions and chat completions
- `lm_eval/models/anthropic_llms.py` — Anthropic API integration
- `lm_eval/models/huggingface.py` — HuggingFace Transformers (local, not API-based)
