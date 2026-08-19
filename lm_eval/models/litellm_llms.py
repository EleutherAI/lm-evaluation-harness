"""LiteLLM AI Gateway model backend for lm-evaluation-harness.

Provides a unified interface to 100+ LLM providers through the LiteLLM SDK.

Usage::

    lm_eval --model litellm \
        --model_args model=anthropic/claude-3-sonnet \
        --tasks hellaswag \
        --num_fewshot 0 \
        --apply_chat_template

See https://docs.litellm.ai/docs/providers for supported providers.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion


if TYPE_CHECKING:
    import asyncio


eval_logger = logging.getLogger(__name__)


@register_model("litellm", "litellm-chat", "litellm-chat-completions")
class LiteLLMChatCompletion(LocalChatCompletion):
    """LiteLLM chat completion model backend.

    Uses ``litellm.completion()`` to route requests to 100+ LLM providers.
    LiteLLM reads provider API keys from environment variables automatically
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY, etc.).

    Model identifiers use provider prefixes::

        gpt-4o                              # OpenAI
        anthropic/claude-3-sonnet           # Anthropic
        gemini/gemini-pro                   # Google
        bedrock/anthropic.claude-3-sonnet   # AWS Bedrock
        azure/my-deployment                 # Azure OpenAI
    """

    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=None,
        **kwargs,
    ):
        try:
            import litellm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "attempted to use 'litellm' LM type, but package `litellm` "
                "is not installed. Please install via "
                "`pip install 'lm-eval[litellm]'` or `pip install litellm`",
            ) from None

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        self._litellm = litellm
        # Drop provider-unsupported params (e.g. seed on Azure AI) instead of erroring
        self._litellm.drop_params = True

    def model_call(
        self,
        messages,
        *,
        generate=True,
        gen_kwargs=None,
        **kwargs,
    ) -> dict | None:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            payload = self._create_payload(
                self.create_message(messages),
                generate=generate,
                gen_kwargs=gen_kwargs,
                seed=self._seed,
                eos=self.eos_string,
                **kwargs,
            )
            response = self._litellm.completion(**payload)
            return response.model_dump()
        except Exception as e:
            eval_logger.error(
                "LiteLLM completion failed for model %s: %s", self.model, e
            )
            raise

    async def amodel_call(
        self,
        session,
        sem: asyncio.Semaphore,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens: list[int] | None = None,
        gen_kwargs: dict | None = None,
        **kwargs,
    ) -> list[str] | list[tuple[float, bool]] | None:
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()
        try:
            response = await self._litellm.acompletion(**payload)
            outputs = response.model_dump()
            answers = (
                self.parse_generations(outputs=outputs)
                if generate
                else self.parse_logprobs(
                    outputs=outputs, tokens=messages, ctxlens=ctxlens
                )
            )
            if cache_keys:
                for res, cache in zip(answers, cache_keys, strict=True):
                    if res is not None:
                        self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except BaseException as e:
            eval_logger.error("LiteLLM async completion failed: %r, retrying.", e)
            raise
        finally:
            if acquired:
                sem.release()

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for LiteLLM chat completions. "
            "Most providers do not return logprobs via the chat API. "
            "Use generate_until tasks (e.g. --tasks gsm8k) instead."
        )

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError(
            "Rolling loglikelihood is not supported for LiteLLM chat "
            "completions. Use generate_until tasks instead."
        )
