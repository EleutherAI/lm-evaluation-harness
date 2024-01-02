import random

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("dummy")
class DummyLM(LM):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests):
        res = []

        for _ in requests:
            res.append((-random.random(), False))

        return res

    def generate_until(self, requests):
        res = []

        for ctx, _ in requests:
            res.append("lol")
            assert ctx.strip() != ""

        return res

    def loglikelihood_rolling(self, requests):
        res = []

        for _ in requests:
            res.append(-random.random())

        return res
