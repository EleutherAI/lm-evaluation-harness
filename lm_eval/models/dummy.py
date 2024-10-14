import random

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("dummy")
class DummyLM(LM):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append((-random.random(), False))

        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            assert request.arguments[0].strip() != ""

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res
