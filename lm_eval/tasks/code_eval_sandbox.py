"""External sandbox client shared by Python code-execution benchmarks."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from statistics import fmean
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


EXECUTOR_ENV = "LM_EVAL_PYTHON_EXECUTOR"
EXECUTOR_TIMEOUT_ENV = "LM_EVAL_PYTHON_EXECUTOR_TIMEOUT"
DEFAULT_EXECUTOR_TIMEOUT_SECONDS = 600


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        raise ValueError("pass@k requires at least one completion")
    if not 0 <= c <= n:
        raise ValueError(f"passing completion count must be in [0, {n}], got {c}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def _executor_command() -> list[str]:
    raw_command = os.environ.get(EXECUTOR_ENV)
    if not raw_command:
        raise RuntimeError(
            "Python functional-correctness metrics execute untrusted generated "
            f"programs. Configure {EXECUTOR_ENV} with a sandboxed JSONL executor "
            "command; there is no local fallback."
        )
    command = shlex.split(raw_command)
    if not command:
        raise RuntimeError(f"{EXECUTOR_ENV} did not contain an executable command")
    return command


def _executor_timeout() -> int:
    raw_timeout = os.environ.get(
        EXECUTOR_TIMEOUT_ENV, str(DEFAULT_EXECUTOR_TIMEOUT_SECONDS)
    )
    try:
        timeout = int(raw_timeout)
    except ValueError as error:
        raise ValueError(f"{EXECUTOR_TIMEOUT_ENV} must be an integer") from error
    if timeout <= 0:
        raise ValueError(f"{EXECUTOR_TIMEOUT_ENV} must be positive")
    return timeout


def _result_passed(result: dict[str, Any], index: int) -> bool:
    if result.get("id") != index:
        raise RuntimeError(
            f"Python executor returned id {result.get('id')!r}; expected {index}"
        )
    if "passed" in result:
        if not isinstance(result["passed"], bool):
            raise TypeError("Python executor 'passed' values must be booleans")
        return result["passed"]
    if "status" in result and "exit_code" in result:
        return result["status"] == "OK" and result["exit_code"] == 0
    raise RuntimeError(
        "Python executor results require either 'passed', or both 'status' "
        "and 'exit_code'"
    )


def run_programs(programs: Sequence[str]) -> list[bool]:
    requests = [
        {
            "schema_version": 1,
            "id": index,
            "language": "python",
            "program": program,
        }
        for index, program in enumerate(programs)
    ]
    stdin = "".join(json.dumps(request) + "\n" for request in requests)
    completed = subprocess.run(  # noqa: S603 - explicit operator-provided sandbox
        _executor_command(),
        input=stdin,
        text=True,
        capture_output=True,
        timeout=_executor_timeout(),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "Python sandbox executor exited with "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != len(requests):
        raise RuntimeError(
            f"Python executor returned {len(lines)} results for {len(requests)} requests"
        )
    outcomes = []
    for index, line in enumerate(lines):
        try:
            result = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Python executor returned invalid JSON at index {index}"
            ) from error
        if not isinstance(result, dict):
            raise TypeError("Python executor results must be JSON objects")
        outcomes.append(_result_passed(result, index))
    return outcomes


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: int | list[int],
    *,
    predictions_include_tests: bool = False,
) -> dict[str, float]:
    if isinstance(k, int):
        k = [k]
    if len(references) != len(predictions):
        raise ValueError("references and predictions must contain the same problems")

    programs: list[str] = []
    group_sizes = []
    for reference, completions in zip(references, predictions, strict=True):
        group_sizes.append(len(completions))
        programs.extend(
            completion if predictions_include_tests else f"{completion}\n{reference}"
            for completion in completions
        )
    outcomes = run_programs(programs)

    per_k: dict[int, list[float]] = {value: [] for value in k}
    offset = 0
    for size in group_sizes:
        problem_outcomes = outcomes[offset : offset + size]
        offset += size
        correct = sum(problem_outcomes)
        for value in k:
            per_k[value].append(estimate_pass_at_k(size, correct, value))
    return {f"pass@{value}": fmean(scores) for value, scores in per_k.items()}
