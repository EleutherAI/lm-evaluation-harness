import os
import types

import pytest

from lm_eval.evaluator_utils import validate_task_safety


def _task(unsafe):
    return types.SimpleNamespace(UNSAFE_CODE=unsafe)


SAFE = {"safe_task": _task(False)}
UNSAFE = {"humaneval": _task(True)}


def test_safe_tasks_pass_without_flags():
    validate_task_safety(SAFE, confirm_run_unsafe_code=False)


def test_unsafe_task_requires_confirmation(monkeypatch):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    with pytest.raises(ValueError, match="confirm_run_unsafe_code=True"):
        validate_task_safety(UNSAFE, confirm_run_unsafe_code=False)


def test_unsafe_task_requires_env_var_even_when_confirmed(monkeypatch):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    with pytest.raises(ValueError, match="HF_ALLOW_CODE_EVAL"):
        validate_task_safety(UNSAFE, confirm_run_unsafe_code=True)


def test_unsafe_task_passes_with_both_gates(monkeypatch):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    validate_task_safety(UNSAFE, confirm_run_unsafe_code=True)


def test_mixed_tasks_fail_on_the_unsafe_one(monkeypatch):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    mixed = {**SAFE, "humaneval": _task(True)}
    with pytest.raises(ValueError, match="humaneval"):
        validate_task_safety(mixed, confirm_run_unsafe_code=True)


def test_env_var_gate_fires_before_generation_not_at_metric_time(monkeypatch):
    """The point of the early check: the error surfaces pre-generation,
    and the message says so (the metric enforces it post-generation)."""
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    with pytest.raises(ValueError, match="after all generation has finished"):
        validate_task_safety(UNSAFE, confirm_run_unsafe_code=True)
