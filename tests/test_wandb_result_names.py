import copy

import numpy as np
import pytest

from lm_eval.loggers.wandb_logger import WandbLogger


def make_logger(results):
    logger = WandbLogger.__new__(WandbLogger)
    logger.post_init({"results": results})
    return logger


@pytest.mark.parametrize("task_name", ["plain", "suite/leaf", "suite/nested/leaf"])
def test_task_names_preserve_summary_and_metric_keys(task_name):
    results = {
        task_name: {
            "name": task_name,
            "alias": "Readable name",
            "acc,none": 0.75,
            "acc_stderr,none": "N/A",
        }
    }
    original = copy.deepcopy(results)
    logger = make_logger(results)
    expected = (
        {
            f"{task_name}/name": task_name,
            f"{task_name}/alias": "Readable name",
            f"{task_name}/acc_stderr": "N/A",
        },
        {f"{task_name}/acc": 0.75},
    )

    assert logger._sanitize_results_dict() == expected
    assert logger._sanitize_results_dict() == expected
    assert results == original
    assert logger.results["results"] == original


@pytest.mark.parametrize("metric_name", ["notes/source", "acc_stderr,filter/strict"])
def test_string_metric_names_can_contain_slashes(metric_name):
    logger = make_logger({"task": {metric_name: "N/A", "acc,none": 1.0}})

    assert logger._sanitize_results_dict() == (
        {f"task/{metric_name}": "N/A"},
        {"task/acc": 1.0},
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_flat_metric_key_does_not_replace_another_task(reverse):
    tasks = [("a", {"score": 1.0}), ("a/score", {"other": 2.0})]
    if reverse:
        tasks.reverse()
    logger = make_logger(dict(tasks))

    assert logger._sanitize_results_dict() == (
        {},
        {"a/score": 1.0, "a/score/other": 2.0},
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_existing_flat_key_collision_order_is_preserved(reverse):
    tasks = [("a", {"b/c": 1.0}), ("a/b", {"c": 2.0})]
    if reverse:
        tasks.reverse()
    logger = make_logger(dict(tasks))

    assert logger._sanitize_results_dict() == (
        {},
        {"a/b/c": 1.0 if reverse else 2.0},
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_none_suffix_collision_precedence_is_preserved(reverse):
    metrics = [("acc,none", 0.75), ("acc", 0.25)]
    if reverse:
        metrics.reverse()
    logger = make_logger({"task": dict(metrics)})

    assert logger._sanitize_results_dict() == ({}, {"task/acc": 0.75})


def test_non_string_values_are_preserved():
    metrics = {
        "missing": None,
        "flag": True,
        "counts": [1, 2],
        "nested": {"value": 3},
        "numpy_value": np.float32(0.5),
    }
    logger = make_logger({"task": metrics})

    assert logger._sanitize_results_dict() == (
        {},
        {f"task/{key}": value for key, value in metrics.items()},
    )


def test_empty_results():
    assert make_logger({})._sanitize_results_dict() == ({}, {})
