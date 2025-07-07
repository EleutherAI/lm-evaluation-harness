from collections import Counter
from typing import Iterable, TypeVar

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


T = TypeVar("T")
# TODO: implement "arg_max" filter. either it should take in an arbitrary "scoring"/reward function
# that takes an input and returns a scalar and then should select the max reward,
# or should implement different filters for different ways of handling a reward model's inference.


@register_filter("take_first")
class TakeFirstFilter(Filter):
    def apply(self, resps: Iterable[list[T]], docs: list[dict]) -> Iterable[list[T]]:
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """
        return map(lambda r: [r[0]], resps)


@register_filter("take_first_k")
class TakeKFilter(Filter):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop("k")
        super().__init__(**kwargs)

    def apply(self, resps: Iterable[list[T]], docs: list[dict]) -> Iterable[list[T]]:
        # need resp to be subscriptable to check below
        resps = list(resps)
        # check we have at least k responses per doc, else we can't take the first k
        assert len(resps[0]) >= self.k, (
            f"Need at least {self.k} responses per doc to take first {self.k}, but got {len(resps[0])} only! Please increase TaskConfig.repeats ."
        )
        return map(lambda r: r[: self.k], resps)


@register_filter("majority_vote")
class MajorityVoteFilter(Filter):
    def apply(self, resps: Iterable[list[T]], docs: list[dict]) -> Iterable[list[T]]:
        """
        Each entry of `resps` is a list of model responses.
        We select the response that occurs most frequently in each entry of `resps`.
        """

        def select_majority(resp):
            counts = Counter(resp)
            vote = counts.most_common(1)[0][0]
            return vote

        return map(lambda r: [select_majority(r)], resps)
