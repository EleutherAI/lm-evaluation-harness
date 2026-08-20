import json
import subprocess

import pytest

from lm_eval.tasks import code_eval_sandbox
from lm_eval.tasks.cruxeval import utils as cruxeval_utils
from lm_eval.tasks.humaneval import utils as humaneval_utils
from lm_eval.tasks.mbpp import utils as mbpp_utils


def test_sandbox_client_combines_candidate_and_tests(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["requests"] = [
            json.loads(line) for line in kwargs["input"].splitlines()
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"id": 0, "passed": true}\n{"id": 1, "passed": false}\n',
            stderr="",
        )

    monkeypatch.setenv(code_eval_sandbox.EXECUTOR_ENV, "sandbox-run --jsonl")
    monkeypatch.setattr(code_eval_sandbox.subprocess, "run", fake_run)

    scores = code_eval_sandbox.pass_at_k(
        ["assert candidate(1) == 2"],
        [["def candidate(x): return x + 1", "def candidate(x): return x"]],
        [1, 2],
    )

    assert captured["command"] == ["sandbox-run", "--jsonl"]
    assert captured["requests"][0]["program"].endswith("\nassert candidate(1) == 2")
    assert scores == {"pass@1": 0.5, "pass@2": 1.0}


def test_sandbox_has_no_local_fallback(monkeypatch):
    monkeypatch.delenv(code_eval_sandbox.EXECUTOR_ENV, raising=False)

    with pytest.raises(RuntimeError, match=code_eval_sandbox.EXECUTOR_ENV):
        code_eval_sandbox.pass_at_k(["assert True"], [["pass"]], [1])


def test_task_wrappers_all_delegate_to_the_external_client(monkeypatch):
    calls = []

    def fake_pass_at_k(references, predictions, k, **kwargs):
        calls.append((references, predictions, k, kwargs))
        return {"pass@1": 0.75}

    monkeypatch.setattr(humaneval_utils, "sandbox_pass_at_k", fake_pass_at_k)
    monkeypatch.setattr(mbpp_utils, "pass_at_k", fake_pass_at_k)
    monkeypatch.setattr(cruxeval_utils, "sandbox_pass_at_k", fake_pass_at_k)

    assert humaneval_utils.pass_at_k(["tests"], [["code"]], [1]) == {"pass@1": 0.75}
    assert mbpp_utils.pass_at_1(["tests"], [["code"]]) == 0.75
    assert cruxeval_utils.pass_at_k(["ignored"], [["complete program"]], [1]) == {
        "pass@1": 0.75
    }
    assert calls[-1][-1] == {"predictions_include_tests": True}
