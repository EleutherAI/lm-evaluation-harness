from .aggregation import *
from .metric import *

from lm_eval.api.metrics import bootstrap_stderr, mean_stderr, acc_all_stderr
from lm_eval.api.register import (
    metric_registry,
    aggregation_registry,
    higher_is_better_registry,
    output_type_registry,
    default_aggregation_registry,
)

METRIC_REGISTRY = metric_registry
OUTPUT_TYPE_REGISTRY = output_type_registry
AGGREGATION_REGISTRY = aggregation_registry
DEFAULT_AGGREGATION_REGISTRY = default_aggregation_registry
HIGHER_IS_BETTER_REGISTRY = higher_is_better_registry

DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": [
        "acc",
    ],
    "greedy_until": ["exact_match"],
}


def get_metric(name):

    try:
        return METRIC_REGISTRY[name]
    except KeyError:
        # TODO: change this print to logging?
        print(
            f"Could not find registered metric '{name}' in lm-eval, \
searching in HF Evaluate library..."
        )
        try:
            import evaluate

            metric_object = evaluate.load(name)
            return metric_object.compute
        except Exception:
            raise Warning(
                "{} not found in the evaluate library!".format(name),
                "Please check https://huggingface.co/evaluate-metric",
            )


def get_aggregation(name):

    try:
        return AGGREGATION_REGISTRY[name]
    except KeyError:
        raise Warning(
            "{} not a registered aggregation metric!".format(name),
        )


def stderr_for_metric(metric, bootstrap_iters):
    bootstrappable = [
        "median",
        "matthews_corrcoef",
        "f1_score",
        "perplexity",
        "bleu",
        "chrf",
        "ter",
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(
            METRIC_REGISTRY[metric], x, iters=bootstrap_iters
        )

    stderr = {"mean": mean_stderr, "acc_all": acc_all_stderr}

    return stderr.get(metric, None)
