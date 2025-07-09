import os
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.models.openai_completions import LocalChatCompletion


eval_logger = logging.getLogger(__name__)


@register_model("mistral-completions")
class MistralCompletionsAPI(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.mistral.ai/v1/chat/completions",
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
        if "batch_size" in kwargs and int(kwargs["batch_size"]) > 1:
            eval_logger.warning(
                "Mistral API does not support batching, setting --batch_size=1"
            )
            self._batch_size = 1
        if "model_name" not in kwargs:
            raise ValueError(
                "MistralCompletionsAPI requires a 'model_name' argument to be set."
            )
        self.model_name = kwargs["model_name"]

    @cached_property
    def eos_string(self) -> Optional[str]:
        return "string" #! NOT SURE (default value in API)

    @cached_property
    def api_key(self):
        key = os.environ.get("MISTRAL_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the MISTRAL_API_KEY env variable."
            )
        return key

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def create_message(
        self,
        messages: Union[List[List[int]], List[str]],
        generate=False,
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        return [{
            "role": "user",
            "content": msg
        } for msg in messages]

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[str]],
        generate=True,
        gen_kwargs: dict = None,
        seed: int = 1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", [self.eos_string]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        
        return {
            "model": self.model_name,
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": False,
            "stop": stop,
            "random_seed": seed,
            "messages": messages,
            "n": 1,
            "prompt_mode": None,
            "safe_prompt": False
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        # NOTE: this method copied from OpenAICompletionsAPI, just adapted for Mistral API
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]
            res = res + tmp
        return res


