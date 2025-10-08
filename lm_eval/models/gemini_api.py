from __future__ import annotations

import logging
import os
from functools import cached_property

from lm_eval.api.registry import register_model

from .openai_completions import OpenAIChatCompletion
from .utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("gemini-openai")
class GeminiOpenAI(OpenAIChatCompletion):
    def __init__(
        self,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
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

    def _create_payload(
        self,
        *args,
        **kwargs,
    ) -> dict:
        _res = super()._create_payload(*args, **kwargs)
        _res.pop("seed", None)
        stop = handle_stop_sequences(
            kwargs["gen_kwargs"].pop("until", None), kwargs["eos"]
        )
        if len(stop) > 0:
            eval_logger.warning(
                "Gemini API does not support multiple stop sequences. Using first sequence."
            )
            stop = stop[0]
        else:
            stop = ""
        assert isinstance(stop, str)
        _res["stop"] = stop
        return _res

    @cached_property
    def api_key(self):
        key = os.environ.get("GEMINI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `GEMINI_API_KEY` environment variable."
            )
        return key
