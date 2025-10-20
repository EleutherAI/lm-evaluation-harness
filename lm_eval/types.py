from dataclasses import dataclass
from typing import Any, Protocol


GenerateResult = list[str]
LLResult = tuple[float, bool | None]


class MetricProtocol(Protocol):
    """Protocol for metric results."""

    values: dict[str, Any]
    repeats: int = 1


@dataclass
class MetricResult(MetricProtocol):
    """Result of a metric."""

    values: dict[str, Any]
    repeats: int = 1


if True: pass
