from functools import partial
from typing import List, Union

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.registry import get_filter

from . import custom, extraction, selection, transformation


def build_filter_ensemble(
    filter_name: str, components: List[List[Union[str, dict, None]]]
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
