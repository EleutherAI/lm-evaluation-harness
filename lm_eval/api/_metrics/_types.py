from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import Protocol, TypeVar


if TYPE_CHECKING:
    from collections.abc import Sequence


_T = TypeVar("_T")
_K = TypeVar("_K")


class MetricFn(Protocol[_T]):
    """Callable that computes a per-sample metric value."""

    def __call__(
        self, references: Any, predictions: Any, **kwargs: Any
    ) -> _T | dict[str, list[_T]]: ...


class ReductionFn(Protocol[_T, _K]):
    """Callable that reduces per-repeat scores into one value per document."""

    def __call__(
        self, references: Any, predictions: Sequence[_T]
    ) -> _K | _T | dict[str, _K] | dict[str, _T]: ...


class AggregationFn(Protocol[_K]):
    """Callable that aggregates per-document values into a corpus-level float."""

    def __call__(self, values: Sequence[_K]) -> float: ...
