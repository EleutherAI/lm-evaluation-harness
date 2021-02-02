import random
from lm_eval.base import LM


class DummyLM(LM):
    def __init__(self):
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string):
        return cls()

    def loglikelihood(self, requests):
        res = []
        
        for _ in requests:
            res.append((-random.random(), False))

        return res
    
    def greedy_until(self, requests):
        # TODO: implement
        pass
