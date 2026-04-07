from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, List, Union

from lm_eval.api.instance import Instance


class Filter(ABC):
    """
    Filter classes operate on a per-task level.
    They take all model outputs (`instance.resps` for all `task.instances`)
    across all instances of a task, and perform operations.
    In a single run, one can configure any number of separate filters or lists of filters.

    """

    def __init__(self, **kwargs) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    @abstractmethod
    def apply(self, resps: Union[List, Iterable], docs: List[dict]) -> Iterable:
        """
        Defines the operation to perform on a list of the `inst.resps` properties of `Instance` objects.
        Should return the list of (filtered) response lists *in the same order as they were input*, e.g.
        if pass in [<inst.resps for instance 0>, <inst.resps for instance 1>] should return
        [<filtered resps for instance 0>, <filtered resps for instance 1>]
        """
        return resps
    

    def apply_wkwargs(self, resps: Union[List, Iterable], docs: List[dict], **kwargs) -> Iterable:
        """
        Same behaviour as `apply` but with the added parsing of keyword arguments, for backward compatibility.

        This function is intended to be used when the model response (or part of it) is produced, for example, in the 
        `tool_calls` or `reasoning` field of a chat generation request.
        By default this will fallback to normal apply, ignoring any additional argument.
        """
        return self.apply(resps, docs)


@dataclass
class FilterEnsemble:
    """
    FilterEnsemble creates a pipeline applying multiple filters.
    Its intended usage is to stack multiple post-processing steps in order.
    `task.apply_filters` should use a list of FilterEnsemble classes that it stores, to apply each
    pipeline separately.
    """

    name: str
    filters: List[Callable[[], Filter]]

    def apply(self, instances: List[Instance]) -> None:
        # Unpack instances
        resps = [inst.resps for inst in instances]
        docs = [inst.doc for inst in instances]
        tool_calls, has_tools = self.get_tool_calls(resps, instances)

        for f in self.filters:
            # apply filters in sequence
            if has_tools:
                resps = f().apply_wkwargs(resps, docs, tool_calls=tool_calls)
            else:
                resps = f().apply(resps, docs)

        # add the end results after filtering to filtered_requests of their respective source instances.
        # has key `self.name`: each FilterEnsemble applied in a given run should use a different name.
        for inst, resp in zip(instances, resps):
            inst.filtered_resps[self.name] = resp

    def get_tool_calls(self, resps: List[str], instances: List[Instance]) -> List[dict]:
         # Check if tool_calls are actually populated (non-empty)
        has_tool_calls = any(inst.tool_calls for inst in instances)

        if has_tool_calls:
            tool_calls = [inst.tool_calls for inst in instances]
            # Verify all tool_calls lists have same length as resps
            if all(len(tc) == len(resp) for tc, resp in zip(tool_calls, resps)):
                # Valid: tool_calls are present and aligned
                pass
            else:
                # Mismatch: fall back to None padding
                tool_calls = [[None] * len(resp) for resp in resps]
        else:
            # No tool_calls present, patch with Nones
            tool_calls = [[None] * len(resp) for resp in resps]

        return tool_calls, has_tool_calls
