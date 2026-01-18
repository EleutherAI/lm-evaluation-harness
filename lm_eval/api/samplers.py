from __future__ import annotations

import logging
from random import Random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, TypeVar

    _T = TypeVar("_T")

eval_logger = logging.getLogger(__name__)


class ContextSampler:
    def __init__(
        self,
        df: Sequence[dict[str, Any]] | None = None,
        *,
        rnd: int | None = None,
        fewshot_indices: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.rnd = Random(rnd)
        self.df = df or []
        self.fewshot_indices = fewshot_indices
        self._loaded = False  # to iterate over fewshot_indices when needed

    def sample(
        self,
        n: int,
        eval_doc: dict[str, Any] | None = None,
        df: Sequence[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Sequence[dict[str, Any]]:
        """
        Sample n documents from the pool.

        Args:
            n: Number of documents to sample
            eval_doc: Optional document to exclude from sampling
            df: Optional list of documents to sample from

        Returns:
            List of sampled documents
        """
        assert n >= 0, "Error: number of samples requested must be >=0"
        if n == 0:
            return []

        if df:
            self.df = df

        assert self.df, "Error: no documents available for sampling."
        res = (
            self.rnd.sample(self.fewshot_docs(), n)
            if not eval_doc
            else self.rm_eval_doc(
                eval_doc, self.rnd.sample(self.fewshot_docs(), n + 1), n
            )
        )
        assert len(res) == n, (
            f"Error: number of fewshot samples returned ({len(res)}) not equal to number requested ({n})."
        )
        return res

    def set_rnd(self, rnd: int | None):
        self.rnd = Random(rnd)
        return self

    def replace_df(self, df: Sequence[dict[str, Any]]):
        self.df = df
        self._loaded = False
        return self

    def fewshot_docs(self):
        """Return cached fewshot docs if available"""
        if self._loaded:
            return self.df
        if self.fewshot_indices and self.df and not self._loaded:
            self.df = [self.df[i] for i in self.fewshot_indices]
        self._loaded = True
        return list(self.df)

    @staticmethod
    def rm_eval_doc(doc: _T, _iter: Iterable[_T], n=None) -> Sequence[_T]:
        return (
            [x for x in _iter if x != doc]
            if n is None
            else [x for x in _iter if x != doc][:n]
        )


class FirstNSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.df), (
            f"Error: number of fewshot samples requested exceeds the {len(self.df)} that are available."
        )
        return self.df[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        raise NotImplementedError


class ManualSampler(ContextSampler):
    def sample(self, n: int, eval_doc=None, df=None, **kwargs):
        """ """
        raise NotImplementedError


SAMPLER_REGISTRY: dict[str, type[ContextSampler]] = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name: str):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        ) from e
