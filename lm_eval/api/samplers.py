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
        docs: Sequence[dict[str, Any]] | None = None,
        *,
        rnd: int | None = None,
        fewshot_indices: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.rnd = Random(rnd)
        self.docs = docs or []
        self.fewshot_indices = fewshot_indices

        if self.fewshot_indices and self.docs:
            self.docs = [self.docs[i] for i in self.fewshot_indices]

    def sample(
        self, n: int, doc: dict[str, Any] | None = None, **kwargs
    ) -> Sequence[dict]:
        """
        Sample n documents from the pool.

        Args:
            n: Number of documents to sample
            doc: Optional document to exclude from sampling

        Returns:
            List of sampled documents
        """
        if n <= 0:
            return []
        return (
            self.rnd.sample(self.docs, n)
            if not doc
            else self.remove_doc(doc, self.rnd.sample(self.docs, n + 1))
        )

    def set_rnd(self, rnd: int | None) -> None:
        self.rnd = Random(rnd)

    @staticmethod
    def remove_doc(doc: _T, _iter: Iterable[_T]) -> list[_T]:
        return [x for x in _iter if x != doc]


class FirstNSampler(ContextSampler):
    def sample(self, n: int, doc=None, **kwargs):
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), (
            f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        )
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n: int, doc=None, **kwargs):
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        raise NotImplementedError


class ManualSampler(ContextSampler):
    def sample(self, n: int, doc=None, **kwargs):
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
