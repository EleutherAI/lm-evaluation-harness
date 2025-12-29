import random
from functools import cached_property

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("dummy")
class DummyLM(LM):
    tokenizer_name = "allenai/Olmo-3-32B-Think"

    def __init__(self, *args, write_out: bool = False, **kwargs) -> None:
        super().__init__()
        self.write_out = write_out

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append((-random.random(), False))
            if self.write_out:
                print(f"context: {request.arguments[0]}")
                print(f"continuation: {request.arguments[1]}")

        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            if self.write_out:
                print(request.arguments[0])
                print(f"gen_kwargs: {request.arguments[0]}")
            assert request.arguments[0].strip() != ""

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
