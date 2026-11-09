from __future__ import annotations

from functools import partial

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.registry import filter_registry, get_filter

from . import custom, extraction, selection, transformation


def build_filter_ensemble(
    filter_name: str,
    components: list[tuple[str, dict[str, str | int | float] | None]],
) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    # create filters given its name in the registry, and add each as a pipeline step
    return FilterEnsemble(
        name=filter_name,
        filters=[
            partial(get_filter(func), **(kwargs or {})) for func, kwargs in components
        ],
    )


__all__ = [
    "custom",
    "extraction",
    "selection",
    "transformation",
    "build_filter_ensemble",
]
