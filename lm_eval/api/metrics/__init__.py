"""Public metrics surface for lm_eval.

Curated re-exports from the metrics subpackage. Import from here
(``from lm_eval.api.metrics import ...``) for the stable public API.
"""

import logging

from ._types import AggregationFn, MetricFn, ReductionFn
from .aggregations import (
    bits_per_byte,
    bypass_agg,
    mean,
    median,
    nanmean,
    perplexity,
    weighted_mean,
    weighted_perplexity,
)
from .corpus import (
    F1,
    MCC,
    BitsPerByte,
    Bleu,
    BytePerplexity,
    Chrf,
    CorpusMetric,
    Likelihood,
    Perplexity,
    Ter,
    WordPerplexity,
)
from .generation import exact_match_fn
from .ll import (
    acc,
    acc_bytes,
    acc_mutual_info_fn,
    acc_norm,
    bpb,
    brier_score,
    bypass,
    choice_logprob,
    choice_logprob_norm,
    choice_prob_norm,
    exact_match_mc,
    logprob_fn,
)
from .metric import METRIC_KEYS, Metric, filter_kwargs, take_first
from .results import LLResults
from .stderr import (
    bootstrap_stderr,
    mean_stderr,
    pooled_sample_stderr,
    pop_stddev,
    sample_stddev,
    stderr_for_metric,
)


eval_logger = logging.getLogger(__name__)

__all__ = [
    "F1",
    "MCC",
    "METRIC_KEYS",
    "AggregationFn",
    "BitsPerByte",
    "Bleu",
    "BytePerplexity",
    "Chrf",
    "CorpusMetric",
    "LLResults",
    "Likelihood",
    "Metric",
    "MetricFn",
    "Perplexity",
    "ReductionFn",
    "Ter",
    "WordPerplexity",
    "acc",
    "acc_bytes",
    "acc_mutual_info_fn",
    "acc_norm",
    "aggregate_subtask_metrics",
    "bits_per_byte",
    "bootstrap_stderr",
    "bpb",
    "brier_score",
    "bypass",
    "bypass_agg",
    "choice_logprob",
    "choice_logprob_norm",
    "choice_prob_norm",
    "exact_match_fn",
    "exact_match_mc",
    "filter_kwargs",
    "logprob_fn",
    "mean",
    "mean_stderr",
    "median",
    "metric_max_over_ground_truths",
    "nanmean",
    "perplexity",
    "pooled_sample_stderr",
    "pop_stddev",
    "sample_stddev",
    "stderr_for_metric",
    "take_first",
    "weighted_mean",
    "weighted_perplexity",
]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    # TODO: does not hold for non-mean aggregations
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum(
        metric * size for metric, size in zip(metrics, sizes, strict=True)
    ) / sum(sizes)
