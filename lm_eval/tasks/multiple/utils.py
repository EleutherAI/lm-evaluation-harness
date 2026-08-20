"""Utilities for the MultiPL-E task family.

Generated programs are untrusted.  This module deliberately has no in-process
execution path: scoring requires an explicitly configured external sandbox
executor in addition to lm-eval's ``--confirm_run_unsafe_code`` gate.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


EXECUTOR_ENV = "LM_EVAL_MULTIPLE_EXECUTOR"
EXECUTOR_TIMEOUT_ENV = "LM_EVAL_MULTIPLE_EXECUTOR_TIMEOUT"
DEFAULT_EXECUTOR_TIMEOUT_SECONDS = 600
SUPPORTED_LANGUAGES = frozenset({"cpp", "java", "js", "php", "rs", "sh"})


def doc_to_text(doc: dict[str, Any]) -> str:
    """Match MultiPL-E's base-model generation prompt normalization."""

    return doc["prompt"].strip()


def build_predictions(
    resps: list[list[str]], docs: list[dict[str, Any]]
) -> list[list[str]]:
    """Build sandbox-executor requests using MultiPL-E's program stitching.

    The official evaluator combines the original (unstripped) prompt, the
    stopped completion, a newline, and the translated tests.  JSON strings are
    used because lm-eval filters and sample loggers expect serializable values.
    """

    predictions: list[list[str]] = []
    for responses, doc in zip(resps, docs, strict=True):
        language = doc["language"]
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported MultiPL-E language: {language!r}")

        predictions.append(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": doc["name"],
                        "language": language,
                        "program": doc["prompt"] + response + "\n" + doc["tests"],
                    },
                    ensure_ascii=False,
                )
                for response in responses
            ]
        )
    return predictions


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator used by the official MultiPL-E scorer."""

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
            "MultiPL-E scoring executes untrusted generated programs. Configure "
            f"{EXECUTOR_ENV} with a sandboxed JSONL executor command and rerun with "
            "--confirm_run_unsafe_code. See lm_eval/tasks/multiple/README.md."
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


def _decode_predictions(predictions: Sequence[str]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        try:
            request = json.loads(prediction)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid serialized MultiPL-E prediction at index {index}"
            ) from error

        required = {"schema_version", "name", "language", "program"}
        if not isinstance(request, dict) or not required.issubset(request):
            raise ValueError(
                f"MultiPL-E prediction at index {index} lacks required fields"
            )
        if request["language"] not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported MultiPL-E language: {request['language']!r}")
        request["id"] = index
        requests.append(request)
    return requests


def _result_passed(result: dict[str, Any], index: int) -> bool:
    if result.get("id") != index:
        raise RuntimeError(
            f"MultiPL-E executor returned id {result.get('id')!r}; expected {index}"
        )
    if "passed" in result:
        if not isinstance(result["passed"], bool):
            raise TypeError("MultiPL-E executor 'passed' values must be booleans")
        return result["passed"]

    # Also accept the result fields emitted by the official MultiPL-E executor.
    if "status" in result and "exit_code" in result:
        return result["status"] == "OK" and result["exit_code"] == 0
    raise RuntimeError(
        "MultiPL-E executor results require either 'passed', or both 'status' "
        "and 'exit_code'"
    )


def _run_executor(requests: Sequence[dict[str, Any]]) -> list[bool]:
    stdin = "".join(
        json.dumps(request, ensure_ascii=False) + "\n" for request in requests
    )
    completed = subprocess.run(  # noqa: S603 - command is an explicit user opt-in
        _executor_command(),
        input=stdin,
        text=True,
        capture_output=True,
        timeout=_executor_timeout(),
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr[-2000:]
        raise RuntimeError(
            f"MultiPL-E sandbox executor exited with {completed.returncode}: {stderr}"
        )

    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(output_lines) != len(requests):
        raise RuntimeError(
            "MultiPL-E executor returned "
            f"{len(output_lines)} results for {len(requests)} requests"
        )

    outcomes = []
    for index, line in enumerate(output_lines):
        try:
            result = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"MultiPL-E executor returned invalid JSON at index {index}"
            ) from error
        if not isinstance(result, dict):
            raise TypeError("MultiPL-E executor results must be JSON objects")
        outcomes.append(_result_passed(result, index))
    return outcomes


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: int | list[int] | None = None,
) -> dict[str, float]:
    """Execute one problem's completions in a configured sandbox and score them."""

    del references  # Tests are already embedded by ``build_predictions``.
    if k is None:
        k = [1]
    elif isinstance(k, int):
        k = [k]

    if len(predictions) != 1:
        raise ValueError(
            "MultiPL-E metrics expect exactly one problem per aggregation item"
        )
    requests = _decode_predictions(predictions[0])
    outcomes = _run_executor(requests)
    n = len(outcomes)
    c = sum(outcomes)
    return {f"pass@{value}": estimate_pass_at_k(n, c, value) for value in k}
