from functools import partial

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.registry import filter_registry, get_filter

from . import custom, extraction, selection, transformation


def build_filter_ensemble(
    filter_name: str, components: list[list[str]]
) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filters = []
    for function, kwargs in components:
        if kwargs is None:
            kwargs = {}
        # create a filter given its name in the registry
        f = partial(get_filter(function), **kwargs)
        # add the filter as a pipeline step
        filters.append(f)

    return FilterEnsemble(name=filter_name, filters=filters)


__all__ = [
    "custom",
    "extraction",
    "selection",
    "transformation",
    "build_filter_ensemble",
]
