import random
from typing import List, Tuple

from lm_eval.api.model import LM


class DummyLM(LM):
    def __init__(self):
        super().__init__()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        res = []
        for _ in requests:
            res.append((-random.random(), False))
        return res

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        res = []
        for _ in requests:
            res.append(-random.random())
        return res

    def greedy_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        res = []
        for ctx, _ in requests:
            res.append("null")
            assert ctx.strip() != ""
        return res
