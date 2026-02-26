from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@register_filter("custom")
class CustomFilter(Filter):
    """Custom filter that applies a custom, user-defined function to the model responses."""

    def __init__(self, **kwargs: Any) -> None:
        self.filter_fn = kwargs.pop("filter_fn")

    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]):
        return self.filter_fn(resps, docs)
