from functools import partial
from typing import Optional, Union

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.registry import get_filter

from . import custom, extraction, selection, transformation


def build_filter_ensemble(
    filter_name: str,
    components: list[tuple[str, Optional[dict[str, Union[str, int, float]]]]],
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
