import logging
from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from typing_extensions import TypeVar

from lm_eval.api.registry import register_metric
from lm_eval.utils import warning_once


eval_logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class CorpusMetric(Generic[_T], ABC):
    """Base class for corpus-level metrics.

    Corpus-level metrics are computed across multiple items (e.g. examples, documents)
    and typically require aggregation of intermediate results (e.g. predictions, targets).
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> _T: ...
    @abstractmethod
    def aggregation(self, items: list[_T]) -> float: ...

    def reduce(self, targets: list, results: list[_T], **kwargs) -> _T:
        if len(results) != 1:
            warning_once(
                eval_logger,
                f"CorpusMetric {self.__class__.__name__} received multiple results; expected only one. Returning the first result.",
            )
        return results[0]


@register_metric(
    metric="brier_score",
    higher_is_better=False,
    output_type=["multiple_choice"],
)
class BrierScore(CorpusMetric):
    """Brier score for multiple choice tasks.

    Computes the mean squared error between predicted probabilities
    (from softmax of log-likelihoods) and one-hot encoded targets.

    Lower scores are better (perfect score = 0.0).
    """

    def __call__(self, items: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
        """Extract target and compute softmax probabilities from log-likelihoods.

        Args:
            result: MCResult containing target index and log-likelihoods

        Returns:
            Tuple of (target_index, probability_distribution)
        """
        return items

    def aggregation(self, items) -> float:
        """Compute mean Brier score across all items.

        Args:
            items: Iterable of (target, probabilities) tuples

        Returns:
            Mean Brier score (mean squared error)
        """
        gold, predictions = list(zip(*items, strict=True))
        bs, num_class = np.array(predictions).shape

        gold = list(gold)
        gold_one_hot = np.eye(num_class)[gold]
        return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))
