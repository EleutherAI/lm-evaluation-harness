import random

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import ResponsesResult


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

        return ResponsesResult(res, 0)

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for ctx, _ in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            assert ctx.strip() != ""

        return ResponsesResult(res, 0)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return ResponsesResult(res, 0)
