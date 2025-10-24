from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    import datasets

GenerateResult = list[str]
LLResult = tuple[float, bool | None]
TaskDataSet = datasets.Dataset | Iterable[dict[str, Any]]
DatasetSplits = dict[str, TaskDataSet]


class MetricProtocol(Protocol):
    """Protocol for metric results."""

    values: dict[str, Any]
    repeats: int = 1


@dataclass
class MetricResult(MetricProtocol):
    """Result of a metric."""

    values: dict[str, Any]
    repeats: int = 1
