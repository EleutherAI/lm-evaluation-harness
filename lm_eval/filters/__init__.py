from functools import partial
from typing import List, Union

from lm_eval.api.filter import FilterEnsemble

from . import extraction, selection, transformation


FILTER_REGISTRY = {
    "take_first": selection.TakeFirstFilter,
    "regex": extraction.RegexFilter,
    "majority_vote": selection.MajorityVoteFilter,
    "take_first_k": selection.TakeKFilter,
    "remove_whitespace": extraction.WhitespaceFilter,
    "lowercase": transformation.LowercaseFilter,
    "uppercase": transformation.UppercaseFilter,
    "map": transformation.MapFilter,
    "multi_choice_regex": extraction.MultiChoiceRegexFilter,
    # TODO: implement this filter. either it should take in an arbitrary "scoring"/reward function
    # that takes an input and returns a scalar and then should select the max reward,
    # or should implement different filters for different ways of handling a reward model's inference.
    # "arg_max": selection.ArgMaxFilter,
}


def get_filter(filter_name: str) -> Union[type, str]:
    if filter_name in FILTER_REGISTRY:
        return FILTER_REGISTRY[filter_name]
    else:
        return filter_name


def build_filter_ensemble(
    filter_name: str, components: List[List[str]]
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
