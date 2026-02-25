from collections.abc import Iterable, Sequence
from typing import Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


# TODO: implement "arg_max" filter. either it should take in an arbitrary "scoring"/reward function
# that takes an input and returns a scalar and then should select the max reward,
# or should implement different filters for different ways of handling a reward model's inference.


@register_filter("noop")
class NoopFilter(Filter):
    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        """Noop — preserve all repeats so downstream scoring can handle them."""
        return resps


@register_filter("take_first")
class TakeFirstFilter(Filter):
    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        """Take the first resp."""
        return next(iter(r) for r in resps)


@register_filter("take_first_k")
class TakeKFilter(Filter):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop("k")

    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        # need resp to be subscriptable to check below
        resps = list(resps)
        # check we have at least k responses per doc, else we can't take the first k
        assert len(resps[0]) >= self.k, (
            f"Need at least {self.k} responses per doc to take first {self.k}, but got {len(resps[0])} only! Please increase TaskConfig.repeats ."
        )
        return (r[: self.k] for r in resps)


@register_filter("majority_vote")
class MajorityVoteFilter(Filter):
    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        """
        Each entry of `resps` is a list of model responses.
        We select the response that occurs most frequently in each entry of `resps`.
        """
        from collections import Counter

        def select_majority(resp):
            counts = Counter(resp)
            return counts.most_common(1)[0][0]

        return (select_majority(r) for r in resps)
