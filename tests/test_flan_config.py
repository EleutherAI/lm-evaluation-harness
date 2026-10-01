from pathlib import Path

from lm_eval.tasks._yaml_loader import load_yaml


FLAN_HELD_IN_CONFIG = (
    Path(__file__).parents[1]
    / "lm_eval"
    / "tasks"
    / "benchmarks"
    / "flan"
    / "flan_held_in.yaml"
)


def test_rte_prompt_names_match_aliases():
    """Every RTE prompt has the unique task name implied by its alias."""
    config = load_yaml(FLAN_HELD_IN_CONFIG, resolve_func=False)
    rte_group = next(item for item in config["task"] if item.get("group") == "rte_flan")
    aliases = [item["task_alias"] for item in rte_group["task"]]
    task_names = [item["task"] for item in rte_group["task"]]

    assert aliases == [f"prompt-{index}" for index in range(9)]
    assert task_names == [f"rte_{alias}" for alias in aliases]
