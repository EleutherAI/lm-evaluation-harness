import logging

# Import _metrics package — registers aggregations first, then all metrics.
# Re-export public types, helpers, and aggregation functions.
from lm_eval.api._metrics._types import (
    AggregationFn,
    MetricFn,
    ReductionFn,
)
from lm_eval.api._metrics.aggregations import (
    bits_per_byte,
    bypass_agg,
    mean,
    median,
    nanmean,
    perplexity,
    weighted_mean,
    weighted_perplexity,
)
from lm_eval.api._metrics.corpus import *
from lm_eval.api._metrics.generation import *
from lm_eval.api._metrics.ll import *
from lm_eval.api._metrics.metric import (
    METRIC_KEYS,
    Metric,
    filter_kwargs,
    take_first,
)
from lm_eval.api._metrics.stderr import (
    _bootstrap_internal,
    _bootstrap_internal_no_mp,
    bootstrap_stderr,
    combined_sample_stderr,
    mean_stderr,
    pooled_sample_stderr,
    pop_stddev,
    sample_stddev,
    stderr_for_metric,
)


eval_logger = logging.getLogger(__name__)


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
