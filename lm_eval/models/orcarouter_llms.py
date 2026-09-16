"""OrcaRouter model backend for lm-evaluation-harness.

Provides a named backend for [OrcaRouter](https://www.orcarouter.ai), the
OpenAI-compatible multi-model routing gateway, mirroring the existing
``OpenAIChatCompletion`` backend.

Usage::

    lm_eval --model orcarouter-chat-completions \
        --model_args model=orcarouter/auto \
        --tasks hellaswag \
        --apply_chat_template

Model identifiers use OrcaRouter's namespaced catalog, e.g. ``orcarouter/auto``
(virtual router), ``openai/gpt-5.5`` or ``anthropic/claude-haiku-4.5``. Set the
API key via the ``ORCAROUTER_API_KEY`` environment variable.
"""

import os
from functools import cached_property

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion


@register_model("orcarouter-chat-completions", "orcarouter")
class OrcaRouterChatCompletion(LocalChatCompletion):
    """OrcaRouter ChatCompletions API backend.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint of OrcaRouter,
    a multi-model routing gateway. Like ``OpenAIChatCompletion``, requests are
    sent with ``messages`` as a list of dicts; no local tokenization is needed
    for ``generate_until`` tasks.

    Set the API key with the ``ORCAROUTER_API_KEY`` environment variable.
    """

    def __init__(
        self,
        base_url="https://api.orcarouter.ai/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("ORCAROUTER_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `ORCAROUTER_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OrcaRouter does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 for more background on this limitation."
        )
