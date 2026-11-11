"""AfroBench groups must aggregate metrics that their tasks report (#4480).

A group whose ``aggregate_metric_list`` names a metric that none of its tasks
report gets no aggregate score for that metric.
"""

from pathlib import Path

import pytest

import lm_eval.tasks
from lm_eval.tasks import TaskManager
from lm_eval.tasks._index import Kind


AFROBENCH = Path(lm_eval.tasks.__file__).parent / "afrobench"
INDEX = TaskManager(include_path=AFROBENCH, include_defaults=False).task_index
GROUPS = sorted(name for name, entry in INDEX.items() if entry.kind is Kind.GROUP)


def _leaf_tasks(name: str) -> set[str]:
    entry = INDEX[name]
    if entry.kind in (Kind.TASK, Kind.PY_TASK):
        return {name}
    if entry.kind is Kind.TAG:
        members = [task for task, other in INDEX.items() if name in other.tags]
    else:
        members = [
            t if isinstance(t, str) else t.get("task")
            for t in entry.cfg.get("task", [])
        ]
    return set().union(*(_leaf_tasks(member) for member in members if member in INDEX))


def _reported_name(metric: str) -> str:
    # The index leaves `!function` metrics unresolved, as a path ending in
    # `module.function`; a task reports such a metric under the function's name.
    return metric.rsplit(".", 1)[-1] if Path(metric).is_absolute() else metric


def test_afrobench_groups_are_indexed():
    assert len(GROUPS) >= 20


@pytest.mark.parametrize("group", GROUPS)
def test_afrobench_group_aggregates_only_metrics_its_tasks_report(group):
    tasks = _leaf_tasks(group)
    reported = {
        _reported_name(metric["metric"])
        for task in tasks
        for metric in INDEX[task].cfg.get("metric_list", [])
    }
    aggregated = {
        metric["metric"] for metric in INDEX[group].cfg.get("aggregate_metric_list", [])
    }

    assert tasks, f"{group} resolves to no tasks"
    assert aggregated <= reported, (
        f"{group} aggregates {sorted(aggregated - reported)}, "
        f"which none of its {len(tasks)} tasks report"
    )
