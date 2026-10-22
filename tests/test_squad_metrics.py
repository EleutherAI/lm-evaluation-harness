import importlib.util
import pathlib
import sys

import pytest

from lm_eval.api import squad_metrics


TASKS_DIR = pathlib.Path(__file__).resolve().parents[1] / "lm_eval" / "tasks"

# Task files that used to import ``transformers.data.metrics.squad_metrics``,
# which was removed in transformers 5.20.
TASK_FILES = [
    "xquad/utils.py",
    "coqa/utils.py",
    "super_glue/record/util.py",
    "scrolls/task.py",
    "catalan_bench/utils.py",
    "catalan_bench/truthfulqa_va/utils.py",
    "spanish_bench/utils.py",
    "galician_bench/utils.py",
    "noreval/norquad/utils.py",
]


def test_normalize_answer():
    assert (
        squad_metrics.normalize_answer("  The Quick, brown fox! ") == "quick brown fox"
    )


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("The Eiffel Tower", "eiffel tower.", 1),
        ("Paris", "Paris, France", 0),
        ("", "", 1),
        ("", "something", 0),
    ],
)
def test_compute_exact(gold, pred, expected):
    assert squad_metrics.compute_exact(gold, pred) == expected


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("Paris, France", "Paris", 2 / 3),
        ("the cat sat", "a cat sat", 1.0),
        ("red", "blue", 0),
        # either side empty: 1 if both are empty, otherwise 0
        ("", "", 1),
        ("", "something", 0),
        ("something", "", 0),
    ],
)
def test_compute_f1(gold, pred, expected):
    assert squad_metrics.compute_f1(gold, pred) == pytest.approx(expected)


@pytest.mark.parametrize("rel_path", TASK_FILES)
def test_task_utils_do_not_need_transformers_squad_metrics(rel_path, monkeypatch):
    if rel_path.startswith("scrolls/"):
        # scrolls still uses transformers for its tokenizer
        pytest.importorskip("transformers")
    # Simulate transformers >= 5.20, where this module no longer exists.
    monkeypatch.setitem(sys.modules, "transformers.data.metrics", None)
    monkeypatch.setitem(sys.modules, "transformers.data.metrics.squad_metrics", None)

    path = TASKS_DIR / rel_path
    name = "_squad_check_" + rel_path.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
