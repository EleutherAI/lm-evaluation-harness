import math
import sys
from collections import defaultdict
from decimal import Decimal, localcontext

import datasets
import pytest
from scipy.special import log_softmax

from lm_eval.api.metrics import perplexity, weighted_perplexity
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator_utils import _compute_task_aggregations


MAX_LOG = math.log(sys.float_info.max)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(
    "loss", [0.0, 1.25, 20.0, 700.0, MAX_LOG, math.nextafter(MAX_LOG, math.inf), 1000.0]
)
def test_perplexity_matches_extended_float_range(loss, weighted):
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(loss).exp())

    if weighted:
        # Powers of two preserve the mean loss exactly, including at the overflow boundary.
        actual = weighted_perplexity([(-loss, 1), (-loss, 1), (-2 * loss, 2)])
    else:
        actual = perplexity([-loss, -loss])

    if math.isinf(expected):
        assert actual == math.inf
    else:
        assert actual == pytest.approx(expected, rel=2e-15)
        assert math.isfinite(actual)
    assert isinstance(actual, float)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("loglikelihood", [-math.inf, math.nan])
def test_perplexity_retains_nonfinite_input_semantics(weighted, loglikelihood):
    actual = (
        weighted_perplexity([(loglikelihood, 1)])
        if weighted
        else perplexity([loglikelihood])
    )
    if math.isnan(loglikelihood):
        assert math.isnan(actual)
    else:
        assert actual == math.inf


def test_zero_total_weight_remains_invalid():
    with pytest.raises(ZeroDivisionError):
        weighted_perplexity([(-1000.0, 0)])


@pytest.mark.parametrize("output_type", ["loglikelihood", "loglikelihood_rolling"])
@pytest.mark.parametrize("loss", [10.0, 1000.0])
@pytest.mark.parametrize("bootstrap_iters", [0, 16])
def test_native_task_aggregation_handles_extreme_finite_loglikelihoods(
    output_type, loss, bootstrap_iters, monkeypatch
):
    monkeypatch.setenv("DISABLE_MULTIPROC", "1")
    data = datasets.DatasetDict(
        {"test": datasets.Dataset.from_dict({"text": ["a", "b"]})}
    )
    metric_names = (
        ["perplexity"]
        if output_type == "loglikelihood"
        else ["word_perplexity", "byte_perplexity", "bits_per_byte"]
    )
    task = ConfigurableTask(
        config={
            "task": "finite_loglikelihood_range",
            "output_type": output_type,
            "test_split": "test",
            "custom_dataset": lambda **kwargs: data,
            "doc_to_text": "",
            "doc_to_target": "text",
            "metric_list": [{"metric": metric} for metric in metric_names],
        }
    )
    # These are finite, normalized log probabilities, even when exp(-logp) overflows.
    logp = float(log_softmax([0.0, -loss])[1])
    assert math.isfinite(logp)
    raw_metrics = defaultdict(list)
    for doc in task.eval_docs:
        response = [(logp, False)] if output_type == "loglikelihood" else [logp]
        for metric, value in task.process_results(doc, response).items():
            raw_metrics[metric, "none"].append(value)

    aggregate, count = _compute_task_aggregations(task, raw_metrics, bootstrap_iters)

    assert count == 2
    for metric in metric_names:
        actual = aggregate[f"{metric},none"]
        if metric == "bits_per_byte":
            assert actual == pytest.approx(-logp / math.log(2))
        elif loss == 1000:
            assert actual == math.inf
        else:
            assert actual == pytest.approx(math.exp(-logp))
    if output_type == "loglikelihood" and loss == 1000 and bootstrap_iters:
        # A standard error of infinite replicates is undefined, as it is for -inf inputs.
        assert math.isnan(aggregate["perplexity_stderr,none"])
