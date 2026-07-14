"""Tests that `run` CLI flags are wired through to `simple_evaluate()` correctly.

Gap (ref EleutherAI/lm-evaluation-harness#1883): the existing CLI tests in
`test_cli_subcommands.py` cover argparse parsing and `EvaluatorConfig` precedence
well, but no test asserts that parsed flags actually reach `simple_evaluate(...)`
with the right keyword values. `test_run_command_execute_basic` mocks
`simple_evaluate` and only checks `assert_called_once()`, so a refactor could
silently drop or mis-map a flag (e.g. `--limit`, `--predict_only`, the
`--seed`->four-random-seed mapping, `--gen_kwargs`, `--num_fewshot`) and every
test would still pass.

These tests run the REAL `EvaluatorConfig.from_cli` mapping and assert on the
kwargs a mocked `simple_evaluate` is called with. They are mock-based and
CPU-only: no model is loaded and nothing is downloaded.
"""

import argparse
from unittest.mock import MagicMock, patch

from lm_eval._cli.run import Run
from lm_eval.config.evaluate_config import EvaluatorConfig


# A minimal, valid `run` invocation; individual tests append the flag under test.
BASE_ARGV = [
    "run",
    "--model",
    "hf",
    "--tasks",
    "arc_easy",
    "--model_args",
    "pretrained=gpt2",
]


def _simple_evaluate_kwargs(extra_argv: list[str]) -> dict:
    """Parse `BASE_ARGV + extra_argv` with the real Run parser, execute the
    command with `simple_evaluate` (and surrounding IO) mocked, and return the
    kwargs `simple_evaluate` was called with.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    run_cmd = Run.create(subparsers)
    args = parser.parse_args(BASE_ARGV + extra_argv)

    fake_results = {
        "results": {"arc_easy": {"acc,none": 0.5}},
        "configs": {"arc_easy": {}},
        "versions": {"arc_easy": 1.0},
        "n-shot": {"arc_easy": 0},
        "config": {"batch_sizes": [1]},
        # present so the log_samples/predict_only path (results.pop("samples")) works
        "samples": {"arc_easy": []},
    }

    with (
        patch("lm_eval.simple_evaluate", return_value=fake_results) as mock_se,
        patch("lm_eval.utils.make_table", return_value=""),
        patch("lm_eval.loggers.EvaluationTracker"),
        # Avoid touching the real task registry; we only care about flag wiring.
        patch.object(EvaluatorConfig, "process_tasks", return_value=MagicMock()),
    ):
        run_cmd._execute(args)

    mock_se.assert_called_once()
    return mock_se.call_args.kwargs


def test_model_reaches_simple_evaluate():
    assert _simple_evaluate_kwargs([])["model"] == "hf"


def test_limit_reaches_simple_evaluate():
    assert _simple_evaluate_kwargs(["--limit", "5"])["limit"] == 5.0


def test_num_fewshot_reaches_simple_evaluate():
    assert _simple_evaluate_kwargs(["--num_fewshot", "3"])["num_fewshot"] == 3


def test_predict_only_reaches_simple_evaluate(tmp_path):
    # --predict_only implies --log_samples, which requires --output_path.
    kwargs = _simple_evaluate_kwargs(["--predict_only", "--output_path", str(tmp_path)])
    assert kwargs["predict_only"] is True


def test_gen_kwargs_reaches_simple_evaluate():
    assert _simple_evaluate_kwargs(["--gen_kwargs", "temperature=0"])["gen_kwargs"] == {
        "temperature": 0
    }


def test_seed_maps_to_four_random_seed_kwargs():
    """`--seed a,b,c,d` must map to (python, numpy, torch, fewshot) seeds in order.

    This ordering is exactly the kind of wiring a refactor can silently break.
    """
    kwargs = _simple_evaluate_kwargs(["--seed", "0,1,2,3"])
    assert kwargs["random_seed"] == 0
    assert kwargs["numpy_random_seed"] == 1
    assert kwargs["torch_random_seed"] == 2
    assert kwargs["fewshot_random_seed"] == 3
