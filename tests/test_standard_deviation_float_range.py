import math
import random
from collections import defaultdict
from decimal import Decimal, localcontext

import datasets
import pytest
from scipy.special import log_softmax

from lm_eval.api.metrics import mean_stderr, pop_stddev, sample_stddev
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator_utils import _compute_task_aggregations


def decimal_stddev(values, ddof):
    with localcontext() as context:
        context.prec = 100
        numbers = [Decimal(value) for value in values]
        average = sum(numbers) / len(numbers)
        variance = sum((value - average) ** 2 for value in numbers) / (
            len(numbers) - ddof
        )
        return float(variance.sqrt())


@pytest.mark.parametrize("scale", [1e-200, 1e-170, 1.0, 1e170, 1e200])
@pytest.mark.parametrize("ddof", [0, 1])
def test_standard_deviation_retains_representable_scaled_results(scale, ddof):
    values = [-3 * scale, -scale, 2 * scale, 6 * scale]
    original = values.copy()
    expected = decimal_stddev(values, ddof)
    actual = (pop_stddev if ddof == 0 else sample_stddev)(values)

    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=3e-15, abs=0)
    assert values == original


@pytest.mark.parametrize("values", [[math.inf, 1.0], [-math.inf, 1.0], [math.nan, 1.0]])
@pytest.mark.parametrize("function", [pop_stddev, sample_stddev, mean_stderr])
def test_nonfinite_observations_keep_undefined_standard_deviation(values, function):
    assert math.isnan(function(values))


@pytest.mark.parametrize("function", [pop_stddev, sample_stddev])
def test_constant_extreme_values_have_zero_spread(function):
    assert function([1e200] * 4) == 0.0
    assert function([1e-200] * 4) == 0.0


def test_insufficient_observations_remain_invalid():
    with pytest.raises(ZeroDivisionError):
        pop_stddev([])
    with pytest.raises(ZeroDivisionError):
        sample_stddev([1.0])


@pytest.mark.parametrize("loss_offset", [0.0, 400.0])
def test_native_perplexity_bootstrap_reports_finite_standard_error(
    loss_offset, monkeypatch
):
    monkeypatch.setenv("DISABLE_MULTIPROC", "1")
    data = datasets.DatasetDict(
        {"test": datasets.Dataset.from_dict({"text": ["a", "b", "c", "d"]})}
    )
    task = ConfigurableTask(
        config={
            "task": "finite_perplexity_uncertainty",
            "output_type": "loglikelihood",
            "test_split": "test",
            "custom_dataset": lambda **kwargs: data,
            "doc_to_text": "",
            "doc_to_target": "text",
            "metric_list": [{"metric": "perplexity"}],
        }
    )
    losses = [loss_offset + loss for loss in [1.0, 2.0, 3.0, 4.0]]
    logprobs = [float(log_softmax([0.0, -loss])[1]) for loss in losses]
    raw_metrics = defaultdict(list)
    for doc, logprob in zip(task.eval_docs, logprobs, strict=True):
        for metric, value in task.process_results(doc, [(logprob, False)]).items():
            raw_metrics[metric, "none"].append(value)

    draws = 32
    rng = random.Random(0)
    reference_bootstrap = [
        math.exp(-sum(rng.choices(logprobs, k=len(logprobs))) / len(logprobs))
        for _ in range(draws)
    ]
    expected_stderr = decimal_stddev(reference_bootstrap, 1)
    aggregate, count = _compute_task_aggregations(task, raw_metrics, draws)

    assert count == len(logprobs)
    assert aggregate["perplexity,none"] == pytest.approx(
        math.exp(-sum(logprobs) / len(logprobs))
    )
    actual_stderr = aggregate["perplexity_stderr,none"]
    assert math.isfinite(actual_stderr)
    assert actual_stderr == pytest.approx(expected_stderr, rel=3e-15, abs=0)
