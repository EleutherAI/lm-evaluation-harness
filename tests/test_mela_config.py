from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


MELA_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks" / "mela"


def test_mela_group_includes_each_defined_task_once():
    """The MELA group contains every language task without duplicates."""
    group_tasks = load_yaml(MELA_DIR / "_mela.yaml", resolve_func=False)["task"]
    defined_tasks = {path.stem for path in MELA_DIR.glob("mela_*.yaml")}

    assert len(group_tasks) == len(set(group_tasks))
    assert set(group_tasks) == defined_tasks
