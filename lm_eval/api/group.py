import abc
from dataclasses import asdict, dataclass
from inspect import getsource
from typing import Any, Callable, List, Optional, Union


@dataclass
class AggMetricConfig(dict):
    metric: Optional[str] = None
    aggregation: Optional[str] = "mean"
    weight_by_size: Optional[str] = False
    # list of filter names which should be incorporated into the aggregated metric.
    filter_list: Optional[Union[str, list]] = "none"

    def __post_init__(self):
        if self.aggregation != "mean" and not callable(self.aggregation):
            raise ValueError(
                f"Currently, 'mean' is the only pre-defined aggregation across groups' subtasks. Got '{self.aggregation}'."
            )

        if isinstance(self.filter_list, str):
            self.filter_list = [self.filter_list]


@dataclass
class GroupConfig(dict):
    group: Optional[str] = None
    group_alias: Optional[str] = None
    task: Optional[Union[str, list]] = None
    aggregate_metric_list: Optional[
        Union[List[AggMetricConfig], AggMetricConfig, dict]
    ] = None
    metadata: Optional[dict] = (
        None  # by default, not used in the code. allows for users to pass arbitrary info to tasks
    )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def __post_init__(self):
        if self.aggregate_metric_list is not None:
            if isinstance(self.aggregate_metric_list, dict):
                self.aggregate_metric_list = [self.aggregate_metric_list]

            self.aggregate_metric_list = [
                AggMetricConfig(**item) if isinstance(item, dict) else item
                for item in self.aggregate_metric_list
            ]

    def to_dict(self, keep_callable: bool = False) -> dict:
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Union[Callable, str], keep_callable=False
    ) -> Union[Callable, str]:
        """Serializes a given function or string.

        If 'keep_callable' is True, the original callable is returned.
        Otherwise, attempts to return the source code of the callable using 'getsource'.
        """
        if keep_callable:
            return value
        else:
            try:
                return getsource(value)
            except (TypeError, OSError):
                return str(value)


class ConfigurableGroup(abc.ABC):
    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        self._config = GroupConfig(**config)

    @property
    def group(self):
        return self._config.group

    @property
    def group_alias(self):
        return self._config.group_alias

    @property
    def version(self):
        return self._config.version

    @property
    def config(self):
        return self._config.to_dict()

    @property
    def group_name(self) -> Any:
        return self._config.group

    def __repr__(self):
        return f"ConfigurableGroup(group={self.group},group_alias={self.group_alias})"
