import json
import os
import time
from functools import cached_property
from typing import Dict, List, Optional, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion


@register_model(
    "yandexgpt-llms",
)
class YandexGPTAPI(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        quota_timeout=1,
        **kwargs,
    ):
        super().__init__(base_url=base_url, **kwargs)

        # quota_timeout is to be used to sleep some time to avoid quota exceed
        try:
            self.quota_timeout = float(quota_timeout)
        except ValueError:
            raise ValueError("`quota_timeout` should be int or float value")

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        **kwargs,
    ) -> dict:
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 1.0)
        stream = gen_kwargs.pop("stream", False)
        ### AVOIDING QUOTA EXCEED
        time.sleep(self.quota_timeout)
        return {
            "modelUri": f"gpt://{self.folder}/{self.model}",
            "completionOptions": {
                "stream": bool(stream),
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": messages,
        }

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]]
    ) -> Union[str, JsonChatStr]:
        """
        chat_history here is a list that looks like:
        [{"role": some_role, "content": some_content}, ...]

        YandexGPT takes the same lists of dicts, but the key "content"
        should be substituted with "text":
        [{"role": some_role, "text": some_content}, ...]
        
        """
        history = []
        for replica in chat_history:
            new_replica = {}
            for key in replica:
                if key == "content":
                    new_replica["text"] = replica[key]
                else:
                    new_replica[key] = replica[key]
            history.extend([new_replica])
        return JsonChatStr(json.dumps(history, ensure_ascii=False))

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"{self.api_key[0]} {self.api_key[1]}",
            "x-folder-id": f"{self.folder}",
        }

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("YANDEXGPT_API_KEY")
        if key is None:
            key = os.environ.get("YANDEXGPT_IAM_TOKEN")
            if key is None:
                raise ValueError(
                    "API key not found. Please set the `YANDEXGPT_API_KEY` or `YANDEXGPT_IAM_TOKEN` environment variable."
                )
            auth = "Bearer"
        else:
            auth = "Api-Key"
        return auth, key

    @cached_property
    def folder(self):
        key = os.environ.get("YANDEXGPT_FOLDER")
        if key is None:
            raise ValueError(
                "FOLDER not found. Please set the `YANDEXGPT_FOLDER` environment variable."
            )
        return key

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            answer_for_sample = out["result"]["alternatives"][0]["message"]["text"]
            res.extend([answer_for_sample])
        return res

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""
