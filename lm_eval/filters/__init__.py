from __future__ import annotations

from functools import partial

from lm_eval.api.filter import FilterEnsemble

from . import custom, extraction, selection, transformation


_filter_name = str
_kwargs = dict[str, str | int | float] | None


def build_filter_ensemble(
    filter_name: str,
    components: list[tuple[_filter_name, _kwargs]],
) -> FilterEnsemble:
    from lm_eval.api.registry import filter_registry, get_filter

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
