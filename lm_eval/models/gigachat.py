"""GigaChat API backend using the official ``gigachat`` Python SDK."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import handle_stop_sequences


if TYPE_CHECKING:
    import asyncio


eval_logger = logging.getLogger(__name__)


@register_model("gigachat", "gigachat-chat")
class GigaChatLM(LocalChatCompletion):
    """Chat-completions backend for models served by the GigaChat API.

    Authentication, token refresh, TLS configuration, and transport retries are
    delegated to the official ``gigachat`` SDK. Its ``GIGACHAT_*`` environment
    variables are supported without additional configuration.
    """

    def __init__(
        self,
        model: str = "GigaChat",
        base_url: str | None = None,
        auth_url: str | None = None,
        credentials: str | None = None,
        scope: str | None = None,
        access_token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        verify_ssl_certs: bool | None = None,
        ca_bundle_file: str | None = None,
        cert_file: str | None = None,
        key_file: str | None = None,
        key_file_password: str | None = None,
        profanity_check: bool | None = None,
        flags: list[str] | None = None,
        timeout: float | None = None,
        client_max_retries: int | None = None,
        retry_backoff_factor: float | None = None,
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs: Any,
    ) -> None:
        try:
            from gigachat import GigaChat
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "attempted to use the 'gigachat' model backend, but the "
                "`gigachat` package is not installed. Install it with "
                "`pip install 'lm-eval[gigachat]'`."
            ) from None

        super().__init__(
            model=model,
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

        client_options = {
            "base_url": base_url,
            "auth_url": auth_url,
            "credentials": credentials,
            "scope": scope,
            "access_token": access_token,
            "user": user,
            "password": password,
            "verify_ssl_certs": verify_ssl_certs,
            "ca_bundle_file": ca_bundle_file,
            "cert_file": cert_file,
            "key_file": key_file,
            "key_file_password": key_file_password,
            "profanity_check": profanity_check,
            "flags": flags,
            "timeout": timeout,
            "max_retries": client_max_retries,
            "retry_backoff_factor": retry_backoff_factor,
        }
        # Omitting unset values lets the SDK read its GIGACHAT_* environment
        # variables and preserves its secure defaults (notably TLS verification).
        self.client = GigaChat(
            model=model,
            **{
                key: value for key, value in client_options.items() if value is not None
            },
        )

    @property
    def eos_string(self) -> None:
        return None

    def _create_payload(
        self,
        messages: list[dict],
        generate: bool = True,
        gen_kwargs: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        if not generate:
            raise NotImplementedError(
                "Loglikelihood is not supported by the GigaChat API."
            )

        gen_kwargs = copy.deepcopy(gen_kwargs or {})
        do_sample = gen_kwargs.pop("do_sample", None)
        temperature = gen_kwargs.get("temperature")
        if do_sample is False or temperature == 0:
            # GigaChat's API does not have a do_sample switch and historically
            # rejects temperature=0. top_p=0 with no repetition penalty is the
            # SDK-supported equivalent of greedy decoding.
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 0.0
            gen_kwargs["repetition_penalty"] = 1.0
        max_tokens = gen_kwargs.pop(
            "max_tokens", gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        )
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), None)
        if stop is None:
            stop = []
        elif isinstance(stop, str):
            stop = [stop]

        payload = {"messages": messages, "model": self.model, **gen_kwargs}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # The API does not expose stop sequences, so they are applied after the
        # response is received and kept out of the SDK request model.
        payload["_lm_eval_stop"] = stop
        return payload

    @staticmethod
    def _response_dict(response: Any, stop: list[str]) -> dict:
        outputs = response.model_dump(by_alias=True)
        for choice in outputs.get("choices", []):
            content = choice.get("message", {}).get("content")
            if content is None:
                continue
            stop_positions = [
                content.find(item) for item in stop if item and item in content
            ]
            if stop_positions:
                choice["message"]["content"] = content[: min(stop_positions)]
        return outputs

    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs) -> dict:
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )
        stop = payload.pop("_lm_eval_stop")
        response = self.client.chat(payload)
        return self._response_dict(response, stop)

    async def amodel_call(
        self,
        session,
        sem: asyncio.Semaphore,
        messages,
        *,
        generate=True,
        cache_keys=None,
        ctxlens=None,
        gen_kwargs=None,
        **kwargs,
    ) -> list[str]:
        payload = self._create_payload(
            self.create_message(messages),
            generate=generate,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )
        stop = payload.pop("_lm_eval_stop")
        acquired = await sem.acquire()
        try:
            outputs = self._response_dict(await self.client.achat(payload), stop)
            answers = self.parse_generations(outputs)
            if cache_keys:
                for answer, cache_key in zip(answers, cache_keys, strict=True):
                    if answer is not None:
                        self.cache_hook.add_partial("generate_until", cache_key, answer)
            return answers
        except BaseException as error:
            eval_logger.error("GigaChat async completion failed: %r", error)
            raise
        finally:
            if acquired:
                sem.release()

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported by the GigaChat API; use a "
            "generate_until task instead."
        )

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError(
            "Rolling loglikelihood is not supported by the GigaChat API; use a "
            "generate_until task instead."
        )
