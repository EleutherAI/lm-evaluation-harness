from dataclasses import dataclass, field
from typing import Any, Protocol

from lm_eval.api.instance import Instance


class MetricInput(Protocol):
    """
    Standard input for all requests across all output types.
    """

    doc: dict[str, Any]
    results: Any
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_norm(self) -> int: ...
    def compute_metric(self, ref, pred) -> Any: ...


@dataclass
class MCMetric(MetricInput):
    doc: dict[str, Any]
    results: list[tuple[str, bool]]
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_instance(cls, instances: list[Instance], filter_key: str):
        return cls(
            doc=instances[0].doc,
            results=list(instance.filtered_resps[filter_key] for instance in instances),
            target=instances[0].target,
            metadata={"filter": filter_key},
        )

    def compute(self):



if __name__ == "__main__":
    yy: MetricInput = MCMetric("a", "b")
