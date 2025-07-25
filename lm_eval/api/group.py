from dataclasses import asdict, dataclass
from inspect import getsource
from typing import Callable, Optional, Union

from datasets.features.pdf import field


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
class GroupConfig:
    group: Optional[str] = None
    group_alias: Optional[str] = None
    task: Union[str, list] = field(default_factory=list)
    aggregate_metric_list: Optional[
        Union[list[AggMetricConfig], AggMetricConfig, dict]
    ] = None
    metadata: Optional[dict] = (
        None  # by default, not used in the code. allows for users to pass arbitrary info to tasks
    )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def __contains__(self, item):
        """Support 'in' operator for dict-like behavior."""
        return hasattr(self, item)

    def get(self, key, default=None):
        """Dict-like get method."""
        return getattr(self, key, default)

    def __hash__(self):
        """Make GroupConfig hashable based on group name."""
        return hash(self.group)

    def __eq__(self, other):
        """Equality comparison based on group name."""
        if not isinstance(other, GroupConfig):
            return False
        return self.group == other.group

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

    @property
    def version(self) -> str:
        """Returns the version of the group configuration."""
        return self.metadata.get("version", "1.0")

    def __repr__(self):
        return f"GroupConfig(group={self.group},group_alias={self.group_alias})"
