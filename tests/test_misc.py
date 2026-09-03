import random

import pytest

import lm_eval.api.metrics as metrics
from lm_eval.utils import make_table


def test_bootstrapping():
    random.seed(42)
    arr = [random.random() for _ in range(1000)]
    expected = metrics.mean_stderr(arr)
    bootstrapped = metrics.bootstrap_stderr(metrics.mean, arr, iters=100000)

    assert bootstrapped == pytest.approx(expected, abs=1e-4)


def _table_result_dict(task_metrics: dict) -> dict:
    return {
        "results": {"t1": {"alias": "t1", **task_metrics}},
        "versions": {"t1": 1},
        "n-shot": {"t1": 0},
        "higher_is_better": {"t1": {"acc": True}},
        "group_subtasks": {},
    }


def test_make_table_renders_a_boundary_bound_next_to_a_zero_stderr():
    table = make_table(
        _table_result_dict(
            {
                "acc,none": 0.0,
                "acc_stderr,none": 0.0,
                "acc_boundary_ci95,none": (0.0, 0.13319225093904846),
            }
        )
    )

    # The bound replaces a bare "± 0.0000", which reads as infinite precision.
    assert "<= 0.1332" in table
    # The interval must not be rendered as a metric row of its own.
    assert "boundary_ci95" not in table


def test_make_table_renders_the_lower_bound_at_a_saturated_score():
    table = make_table(
        _table_result_dict(
            {
                "acc,none": 1.0,
                "acc_stderr,none": 0.0,
                "acc_boundary_ci95,none": (0.8668077490609515, 1.0),
            }
        )
    )

    assert ">= 0.8668" in table


def test_make_table_is_unchanged_without_a_boundary_interval():
    table = make_table(_table_result_dict({"acc,none": 0.5, "acc_stderr,none": 0.1}))

    # pytablewriter renders a purely numeric Stderr column as a number; the point
    # is that no bound is appended.
    assert "0.1" in table
    assert "<=" not in table
    assert ">=" not in table
