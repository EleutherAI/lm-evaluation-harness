# REMINDER: this code needs to be rewritten for the new framework. Remove this comment when the code is fully converted.

from lm_eval.base import LM
from . import MODEL_REGISTRY


@MODEL_REGISTRY.register("dummy")
class DummyLM(LM):

    def loglikelihood(self, context, continuation):
        return 0.0
