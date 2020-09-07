from lm_eval.base import LM
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register("dummy")
class DummyLM(LM):

    def generate(self, context, max_gen_length):
        return "lol"

    def loglikelihood(self, context, continuation):
        return 0.0
