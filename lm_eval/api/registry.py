import os
import evaluate
from lm_eval.api.model import LM
from lm_eval.logger import eval_logger

MODEL_REGISTRY = {}


def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, LM
            ), f"Model '{name}' ({cls.__name__}) must extend LM class"

            assert (
                name not in MODEL_REGISTRY
            ), f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def get_model(model_name):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load model '{model_name}', but no model for this name found! Supported model names: {', '.join(MODEL_REGISTRY.keys())}"
        )


TASK_REGISTRY = {}
GROUP_REGISTRY = {}
ALL_TASKS = set()
func2task_index = {}


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        func2task_index[fn.__name__] = name
        return fn

    return decorate


def register_group(name):
    def decorate(fn):
        func_name = func2task_index[fn.__name__]
        if name in GROUP_REGISTRY:
            GROUP_REGISTRY[name].append(func_name)
        else:
            GROUP_REGISTRY[name] = [func_name]
            ALL_TASKS.add(name)
        return fn

    return decorate


OUTPUT_TYPE_REGISTRY = {}
METRIC_REGISTRY = {}
METRIC_AGGREGATION_REGISTRY = {}
AGGREGATION_REGISTRY = {}
HIGHER_IS_BETTER_REGISTRY = {}

DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


def register_metric(**args):
    # TODO: do we want to enforce a certain interface to registered metrics?
    def decorate(fn):

        assert "metric" in args
        name = args["metric"]

        for key, registry in [
            ("metric", METRIC_REGISTRY),
            ("higher_is_better", HIGHER_IS_BETTER_REGISTRY),
            ("aggregation", METRIC_AGGREGATION_REGISTRY),
        ]:

            if key in args:
                value = args[key]
                assert (
                    value not in registry
                ), f"{key} named '{value}' conflicts with existing registered {key}!"

                if key == "metric":
                    registry[name] = fn
                elif key == "aggregation":
                    registry[name] = AGGREGATION_REGISTRY[value]
                else:
                    registry[name] = value

        return fn

    return decorate


def get_metric(name, hf_evaluate_metric=False):

    if not hf_evaluate_metric:
        if name in METRIC_REGISTRY:
            return METRIC_REGISTRY[name]
        else:
            eval_logger.warning(
                f"Could not find registered metric '{name}' in lm-eval, searching in HF Evaluate library..."
            )

    try:
        metric_object = evaluate.load(name)
        return metric_object.compute
    except Exception:
        eval_logger.error(
            f"{name} not found in the evaluate library! Please check https://huggingface.co/evaluate-metric",
        )


def register_aggregation(name):
    def decorate(fn):
        assert (
            name not in AGGREGATION_REGISTRY
        ), f"aggregation named '{name}' conflicts with existing registered aggregation!"

        AGGREGATION_REGISTRY[name] = fn
        return fn

    return decorate


def get_aggregation(name):

    try:
        return AGGREGATION_REGISTRY[name]
    except KeyError:
        eval_logger.warning(
            "{} not a registered aggregation metric!".format(name),
        )


def get_metric_aggregation(name):

    try:
        return METRIC_AGGREGATION_REGISTRY[name]
    except KeyError:
        eval_logger.warning(
            "{} metric is not assigned a default aggregation!".format(name),
        )


def is_higher_better(metric_name):
    try:
        return HIGHER_IS_BETTER_REGISTRY[metric_name]
    except KeyError:
        eval_logger.warning(
            f"higher_is_better not specified for metric '{metric_name}'!"
        )
