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
4. `headers`: Returns the headers for the API request.

You may also need to override other methods or properties depending on your API's specific requirements.

> [!NOTE]
> Currently loglikelihood and MCQ based tasks (such as MMLU) are only supported for completion endpoints. Not for chat-completion — those that expect a list of dicts — endpoints! Completion APIs which support instruct tuned models can be evaluated with the `--apply_chat_template` option in order to simultaneously evaluate models using a chat template format while still being able to access the model logits needed for loglikelihood-based tasks.

# TemplateAPI Usage Guide

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
   - Default is 30.

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
   - Maximum number of retries for failed API requests.
   - Default is 3.

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
3. Use environment variables for sensitive information like API keys.
4. Properly handle batching and concurrent requests if supported by your API.
