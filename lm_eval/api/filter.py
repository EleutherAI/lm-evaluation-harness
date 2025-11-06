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
    think_tokens: Union[str, dict]

    def think_filter(self, resps):
        if self.think_tokens: 
            start_token = self.think_tokens.get("think_start_token")
            end_token = self.think_tokens.get("think_end_token")
            if end_token:
                processed_resps_list = []
                for instance_resps in resps:
                    new_instance_resps = []
                    for resp_str in instance_resps: 
                        if not isinstance(resp_str, str):
                            new_instance_resps.append(resp_str)
                            continue
                        new_resp = resp_str 
                        end_pos = resp_str.rfind(end_token)
                        if end_pos != -1:
                            new_resp = resp_str[end_pos + len(end_token) :]
                        elif start_token:
                            start_pos = resp_str.find(start_token)
                            if start_pos != -1:
                                new_resp = resp_str[:start_pos]
                        new_instance_resps.append(new_resp)
                    processed_resps_list.append(new_instance_resps)
                
                # 用处理后的 "列表的列表" 覆盖 resps
                resps = processed_resps_list
                return resps
            else:
                return resps
        else:
            return resps

    def apply(self, instances: List[Instance]) -> None:
        resps, docs = zip(*((inst.resps, inst.doc) for inst in instances))
        resps, docs = list(resps), list(docs)

        resps = self.think_filter(resps)

        for f in self.filters:
            # apply filters in sequence
            resps = f().apply(resps, docs)

        # add the end results after filtering to filtered_requests of their respective source instances.
        # has key `self.name`: each FilterEnsemble applied in a given run should use a different name.
        for inst, resp in zip(instances, resps):
            inst.filtered_resps[self.name] = resp