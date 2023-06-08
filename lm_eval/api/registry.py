import os
import evaluate
from lm_eval.api.model import LM

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
    return MODEL_REGISTRY[model_name]


TASK_REGISTRY = {}
GROUP_REGISTRY = {}
func2task_index = {}


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        func2task_index[fn.__name__] = name
        return fn

    return decorate


def register_group(name):
    def decorate(fn):
        # assert (
        #     name not in GROUP_REGISTRY
        # ), f"group named '{name}' conflicts with existing registered group!"

        func_name = func2task_index[fn.__name__]
        if name in GROUP_REGISTRY:
            GROUP_REGISTRY[name].append(func_name)
        else:
            GROUP_REGISTRY[name] = [func_name]
        return fn

    return decorate


AGGREGATION_REGISTRY = {}
DEFAULT_AGGREGATION_REGISTRY = {}
METRIC_REGISTRY = {}
OUTPUT_TYPE_REGISTRY = {}
HIGHER_IS_BETTER_REGISTRY = {}

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


def register_metric(**args):
    # TODO: do we want to enforce a certain interface to registered metrics?
    def decorate(fn):

        assert "metric" in args
        name = args["metric"]

        for key, registry in [
            ("metric", METRIC_REGISTRY),
            ("higher_is_better", HIGHER_IS_BETTER_REGISTRY),
            # ("output_type", OUTPUT_TYPE_REGISTRY),
            ("aggregation", DEFAULT_AGGREGATION_REGISTRY),
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
            metric_object = evaluate.load(name)
            return metric_object.compute
        except Exception:
            raise Warning(
                "{} not found in the evaluate library!".format(name),
                "Please check https://huggingface.co/evaluate-metric",
            )


def register_aggregation(name):
    # TODO: should we enforce a specific interface to aggregation metrics?
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
        raise Warning(
            "{} not a registered aggregation metric!".format(name),
        )
