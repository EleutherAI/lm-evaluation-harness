from collections import Counter

from lm_eval.api.filter import Filter


class TakeFirstFilter(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """
        return map(lambda r: r[0], resps)


class TakeKFilter(Filter):
    def __init__(self, *args, **kwargs) -> None:
        self.k = kwargs.pop("k")

        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # check we have at least k responses per doc, else we can't take the first k
        assert (
            len(resps[0]) >= self.k
        ), f"Need at least {self.k} responses per doc to take first {self.k}, but got {len(resps[0])} only! Please increase TaskConfig.repeats ."
        return map(lambda r: r[: self.k], resps)


class MajorityVoteFilter(Filter):
    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Each entry of `resps` is a list of model responses.
        We select the response that occurs most frequently in each entry of `resps`.
        """

        def select_majority(resp):
            counts = Counter(resp)
            vote = counts.most_common(1)[0][0]
            return vote

        return map(lambda r: [select_majority(r)], resps)
