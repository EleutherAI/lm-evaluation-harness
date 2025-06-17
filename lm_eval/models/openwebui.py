import copy
import json
from typing import Dict, List, Optional, Union
import requests
import os
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalCompletionsAPI
from ollama import Client
from lm_eval.api.model import LM
import logging

try:
    from tenacity import RetryError
except ModuleNotFoundError:
    pass

eval_logger = logging.getLogger(__name__)

@register_model("openwebui")
class openwebui(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        hf_tokenizer=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, hf_tokenizer=hf_tokenizer, **kwargs
        )


    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                #tmp[choices["index"]] = choices["text"]
                tmp[choices["index"]] = choices["message"]["content"]

            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.getenv("OPENWEBUI_KEY")
    
    @property
    def url(self):
        return os.getenv("OPENWEBUI_URL")
    
    @property
    def header(self) -> dict:
        return {'Authorization': f'Bearer {self.api_key}','Content-Type': 'application/json'}
    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        try:
            return super().loglikelihood(requests, list)
        except:
            raise NotImplementedError("Not yet implemented")


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("Not yet implemented")
       
    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)

        if not self.api_key:
            raise ValueError("Not able to retrieve API Key from environment")
        elif not self.url:
            raise ValueError("Not able to retrieve URL address from environment")
        response=None
        prompt = [{'role': 'user', 'content': self.create_message(messages)}]
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                "model": self.model,
                "messages": prompt
            }

            response = requests.post(url=f'{self.url}/api/chat/completions', headers=headers, json=payload)
            if not response:
                print(payload)
                print('Empty response!')
        except Exception as e:
            # Catch any exception that might occur and print an error message
            print(f"An error occurred: {e}")

        response.raise_for_status()
        return response.json()