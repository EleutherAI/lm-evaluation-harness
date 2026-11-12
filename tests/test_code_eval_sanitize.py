import importlib.util
import os
from pathlib import Path

import pytest

_HELPER = (
    Path(__file__).resolve().parents[1]
    / "lm_eval"
    / "tasks"
    / "code_eval_sanitize.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("code_eval_sanitize", _HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_call_is_wrapped_with_the_plain_result_helper():
    helper = _load()
    reference = "def check(candidate):\n    assert candidate(1, 2) == 3\ncheck(add)\n"
    hardened = helper.harden_humaneval_reference(reference)
    assert "check(__lm_eval_wrap(add))" in hardened
    assert "def __lm_eval_to_plain" in hardened


def _exec_and_check(program: str) -> None:
    namespace = {}
    exec(program, namespace, namespace)


def test_always_equal_return_fails_after_hardening():
    helper = _load()
    prediction = (
        "def add(a, b):\n"
        "    class _Any:\n"
        "        def __eq__(self, other):\n"
        "            return True\n"
        "    return _Any()\n"
    )
    reference = helper.harden_humaneval_reference(
        "def check(candidate):\n    assert candidate(1, 2) == 3\ncheck(add)\n"
    )
    with pytest.raises((TypeError, AssertionError)):
        _exec_and_check(prediction + "\n" + reference)

    correct = "def add(a, b):\n    return a + b\n"
    _exec_and_check(correct + "\n" + reference)


def test_mbpp_style_assert_rejects_always_equal_return():
    helper = _load()
    prediction = helper.harden_mbpp_prediction(
        "def add(a, b):\n"
        "    class _Any:\n"
        "        def __eq__(self, other):\n"
        "            return True\n"
        "    return _Any()\n"
    )
    with pytest.raises((TypeError, AssertionError)):
        _exec_and_check(prediction + "\nassert add(1, 2) == 3\n")

    correct = helper.harden_mbpp_prediction("def add(a, b):\n    return a + b\n")
    _exec_and_check(correct + "\nassert add(1, 2) == 3\n")


@pytest.mark.skipif(
    os.environ.get("HF_ALLOW_CODE_EVAL") != "1",
    reason="optional integration through huggingface evaluate.code_eval",
)
def test_code_eval_metric_rejects_always_equal_return():
    evaluate = pytest.importorskip("evaluate")
    helper = _load()
    compute_ = evaluate.load("code_eval")
    doc = {
        "prompt": "def add(a, b):\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }
    reference = f"{doc['test']}\ncheck({doc['entry_point']})"
    cases = {
        "correct": "    return a + b\n",
        "wrong": "    return a - b\n",
        "always_equal": (
            "    class _Any:\n"
            "        def __eq__(self, other):\n"
            "            return True\n"
            "    return _Any()\n"
        ),
    }
    scores = {}
    for name, body in cases.items():
        result = compute_.compute(
            references=helper.harden_humaneval_references([reference]),
            predictions=[[doc["prompt"] + body]],
            k=[1],
        )
        scores[name] = result[0]["pass@1"]
    assert scores == {"correct": 1.0, "wrong": 0.0, "always_equal": 0.0}
