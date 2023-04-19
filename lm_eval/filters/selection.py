from lm_eval.api.filter import Filter

class TakeFirstFilter:

    def __init__(self):
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps):
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """
        return map(lambda r: r[0], resps)