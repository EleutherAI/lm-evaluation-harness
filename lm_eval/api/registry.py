import os
import evaluate
from lm_eval.api.model import LM

import logging

eval_logger = logging.getLogger("lm-eval")

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


METRIC_FUNCTION_REGISTRY = {}
HIGHER_IS_BETTER_REGISTRY = {}

DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [],
    "loglikelihood_rolling": [],
    "multiple_choice": [],
    "generate_until": [],
}


def register_metric(
    metric=None,
    higher_is_better=None,
    output_type=None,
):
    # TODO: do we want to enforce a certain interface to registered metrics?
    def decorate(fn):

        if type(metric) == str:
            metric_list = [metric]
        elif type(metric) == list:
            metric_list = metric

        for _metric in metric_list:
            METRIC_FUNCTION_REGISTRY[_metric] = fn

            if higher_is_better is not None:
                HIGHER_IS_BETTER_REGISTRY[_metric] = higher_is_better

            if output_type is not None:
                if type(output_type) == str:
                    output_type_list = [output_type]
                elif type(output_type) == list:
                    output_type_list = output_type

                for _output_type in output_type_list:
                    DEFAULT_METRIC_REGISTRY[_output_type].append(_metric)

        return fn

    return decorate


def get_metric(name, hf_evaluate_metric=False):

    if not hf_evaluate_metric:
        if name in METRIC_FUNCTION_REGISTRY:
            return METRIC_FUNCTION_REGISTRY[name]
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


def is_higher_better(metric_name):
    try:
        return HIGHER_IS_BETTER_REGISTRY[metric_name]
    except KeyError:
        eval_logger.warning(
            f"higher_is_better not specified for metric '{metric_name}'!"
        )
