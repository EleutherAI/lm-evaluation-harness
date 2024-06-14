from functools import cached_property
from typing import List, Literal, Optional, Union

import requests
from aiohttp import ClientSession
from tenacity import retry, wait_exponential

from lm_eval.api.model import TemplateLM, eval_logger
from lm_eval.models.utils import handle_pad_token


class TemplateAPI(TemplateLM):
    def __init__(
        self,
        model: str,
        base_url: str = None,
        tokenizer: Optional[str] = None,
        tokenizer_backend: Literal["tiktoken", "huggingface"] = "huggingface",
        truncate: bool = False,
        concurrent=50,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        seed: int = 1234,
        max_length: Optional[int] = None,
        custom_prefix_token_id=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url
        self._tokenizer = tokenizer
        self.api_key = None
        self._batch_size = batch_size
        self._truncate = truncate
        self._max_gen_toks = max_gen_toks
        self._seed = seed
        self._max_length = max_length
        self._concurrent = concurrent
        self.tokenizer_backend = tokenizer_backend
        self.custom_prefix_token_id = custom_prefix_token_id

        if self._tokenizer is None:
            self.tokenizer = None
        else:
            if self.tokenizer_backend == "huggingface":
                import transformers  # noqa: E401

                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self._tokenizer if self._tokenizer else self.model
                )
                handle_pad_token(self.tokenizer)
            elif self.tokenizer_backend == "tiktoken":
                if self.base_url:
                    eval_logger.warning(
                        f"Passed `base_url={self.base_url}` but using (OpenAI) Tiktoken tokenizer backend. "
                        "Pass `tokenizer_backend=huggingface` and provide the HF tokenizer name if your model does not use Tiktoken."
                    )

    @cached_property
    def eot_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.tokenizer_backend == "huggingface":
                return self.tokenizer.eos_token_id
            elif self.tokenizer_backend == "tiktoken":
                return self.tokenizer.eot_token_id

    @cached_property
    def prefix_token_id(self) -> Optional[int]:
        if self.tokenizer is None:
            return None
        else:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if self.tokenizer.bos_token_id is not None:
                return self.tokenizer.bos_token_id
            return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> Union[List[str], List[int]]:
        if self.tokenizer is None:
            return list(string)
        else:
            return self.tokenizer.encode(string, **kwargs)

    def _create_payload(self, *kwargs) -> dict:
        return {"model": self.model, **kwargs}

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10))
    def model_call(self, payload: dict) -> dict:
        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()
        return response.json()

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10))
    async def amodel_call(self, messages, **kwargs) -> dict:
        async with ClientSession() as session:
            async with session.post(
                self.base_url, json=self._create_payload(messages, kwargs)
            ) as response:
                response.raise_for_status()
                return response.json()
