import ast
from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


TASKS = Path(__file__).parents[1] / "lm_eval" / "tasks"
IFEVAL = TASKS / "ifeval"
LEADERBOARD_IFEVAL = TASKS / "leaderboard" / "ifeval"

# The only module `leaderboard/ifeval/ifeval.yaml` loads by path. Everything else
# the task scores with is imported from `lm_eval.tasks.ifeval`.
LOCAL_MODULE = "utils.py"


def _absolute_imports(module_path):
    """Every `from X import ...` target in a module, as dotted strings."""
    tree = ast.parse(module_path.read_text())
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    }


def test_leaderboard_ifeval_ships_no_shadow_copies():
    """The leaderboard task must not carry its own copy of the ifeval checkers.

    `leaderboard/ifeval/` was created by copying `ifeval/` and rewiring only
    `utils.py`, which left `instructions.py`, `instructions_registry.py` and
    `instructions_util.py` unreachable. They then rotted unnoticed for two years:
    the live `instructions_util.py` moved off `pkg_resources` (#2471) while the
    copy kept it, so the copy stopped importing at all under setuptools >= 81.
    A copy that nothing imports cannot be caught by any other test.
    """
    shadowed = {
        path.name
        for path in LEADERBOARD_IFEVAL.glob("*.py")
        if path.name != LOCAL_MODULE and (IFEVAL / path.name).exists()
    }
    assert shadowed == set()


def test_leaderboard_ifeval_scores_with_the_canonical_checkers():
    """`utils.py` reaches the checkers through the package, not through siblings."""
    assert "lm_eval.tasks.ifeval" in _absolute_imports(
        LEADERBOARD_IFEVAL / LOCAL_MODULE
    )


def test_leaderboard_ifeval_yaml_only_references_utils():
    """Nothing else in the task directory is loaded, so nothing else belongs there."""
    config = load_yaml(LEADERBOARD_IFEVAL / "ifeval.yaml", resolve_func=False)
    # Left unresolved, a `!function` tag is the module's absolute path plus the
    # function name: ".../leaderboard/ifeval/utils.process_results".
    candidates = [config["process_results"]] + [
        metric.get("aggregation") for metric in config["metric_list"]
    ]
    modules = {
        Path(value.rsplit(".", 1)[0]).with_suffix(".py")
        for value in candidates
        if isinstance(value, str) and Path(value).is_absolute()
    }

    assert modules == {LEADERBOARD_IFEVAL / LOCAL_MODULE}
