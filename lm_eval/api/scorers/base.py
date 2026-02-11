from dataclasses import dataclass, field

from lm_eval.api.filter import FilterEnsemble
from lm_eval.config.metric import Metric


@dataclass
class Scorer:
    name: str
    filter: FilterEnsemble
    metrics: list[Metric]
    _resps: list = field(default_factory=list)

    def apply_filter(self, instances: list) -> None:
        """
        Applies the filter to the task's instances, and stores the results in each instance's `filtered_resps` property.
        """
        self.filter.apply(instances)

    def apply_metrics(self, instances: list) -> dict[str, list]:
        """
        Applies the scorer's metrics to the task's instances, and returns a dictionary of metric results.
        """
        raise NotImplementedError("Scorer.apply metrics not implemented yet")
