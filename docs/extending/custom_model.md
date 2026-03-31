# Custom Model Backend

This guide walks through adding a new model backend to the evaluation harness by subclassing the `LM` class.

!!! tip
    If your model is behind an HTTP API (OpenAI-compatible or custom), see [API Model Backend](api_model.md) instead — it handles retries, batching, and caching for you.

## Setup

Fork the repo and install in development mode:

```sh
git clone https://github.com/<YOUR-USERNAME>/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout -b <model-type>
pip install -e ".[dev]"
```

Create a new file for your model:

```sh
touch lm_eval/models/<my_model_filename>.py
```

!!! warning
    The filename must not shadow package names. For example, `anthropic.py` is disallowed since `anthropic` is a package on PyPI, but `anthropic_llms.py` works fine.

## Interface

All models must subclass `lm_eval.api.model.LM` and implement three methods:

```python
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

class MyCustomLM(LM):

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        ...

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        ...

    def generate_until(self, requests: list[Instance]) -> list[str]:
        ...
```

`Instance` is a generic dataclass defined in `lm_eval.api.instance`:

```python
Instance[InputT, OutputT]
```

Two type aliases are provided for convenience:

- `LLInstance = Instance[LLArgs, list[LLOutput]]` — for loglikelihood requests, where `LLArgs = tuple[str, str]` is `(context, continuation)` and `LLOutput = tuple[float, bool]` is `(logprob, is_greedy)`
- `GenInstance = Instance[GenArgs, list[Completion]]` — for generation requests, where `GenArgs = tuple[Context, GenKwargs]` and `Completion = str`

### Request types

**`generate_until`**

- `Instance.args` → `(input_string, gen_kwargs_dict)`
- Sample text from the LM until a stop sequence or max length
- Return the generated text string

**`loglikelihood`**

- `Instance.args` → `(context_string, continuation_string)`
- Return `(logprob, is_greedy)` — the log probability of the continuation given the context, and whether greedy decoding would produce the continuation

**`loglikelihood_rolling`**

- `Instance.args` → `(input_string,)`
- Return the loglikelihood of the entire input conditioned on the EOT token
- Used for perplexity evaluation

!!! tip "Indexing in loglikelihood"
    LMs take tokens at positions `[0 1 2 ... N]` and output probabilities for position `N+1`. The final target token is not passed to the LM — we want predictions *up to but not past* it:
    ```text
    # inp    0 1 2 3|4 5 6 7 8 9   <- last token deleted by inp[:, :-1]
    # model  \               \
    # logits   1 2 3|4 5 6 7 8 9   <- ctx half tossed out
    # cont_toks      4 5 6 7 8 9
    ```

## Registration

Register your model with a name for CLI access:

```python
from lm_eval.api.registry import register_model

@register_model("my_model", "my_model_alt_name")
class MyCustomLM(LM):
    ...
```

Then import it in `lm_eval/models/__init__.py`.

Now you can use it:

```bash
lm-eval run --model my_model --model_args pretrained=my-model-path --tasks hellaswag
```

## Chat template support

To support `--apply_chat_template`, implement three additional methods:

```python
class MyCustomLM(LM):

    @property
    def tokenizer_name(self) -> str:
        """Return the tokenizer/chat template name (used for cache keying)."""
        ...

    def chat_template(self, chat_template: bool | str = False) -> str:
        """Return the Jinja chat template string.

        Args:
            chat_template: False = no template, True = default, str = named template.
        """
        ...

    def apply_chat_template(self, chat_history: list[dict[str, str]]) -> str:
        """Convert a chat history into a string for the model.

        Args:
            chat_history: List of {"role": "...", "content": "..."} dicts.
        """
        ...
```

The chat history uses the standard `{"role": ..., "content": ...}` format:

```python
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And 3+3?"},  # <- test query
]
```

Without these methods, `--apply_chat_template`, `--fewshot_as_multiturn`, and `--system_instruction` cannot be used.

## Helpful base classes

- **`LM`** — the minimal base class
- **`TemplateLM`** — abstracts commonly-used functions across LM subclasses
- **`HFLM`** (`lm_eval.models.huggingface`) — full HuggingFace implementation, often the easiest to subclass if your model is HF-compatible

## Testing

New model contributions should include tests for all three core methods. See `tests/test_ggml.py` for an example.

## Performance tip

To improve runtime estimates, implement request reordering (process longest inputs first). See `lm_eval.utils.Reorderer` and how `HFLM` uses it.

## Next steps

- [API Model Backend](api_model.md) — for HTTP API-based models
- [Custom Scorers](custom_scorers.md) — implement custom scoring logic for your tasks
- [Custom Metrics & Filters](custom_metrics_and_filters.md) — add new metrics or post-processing filters
