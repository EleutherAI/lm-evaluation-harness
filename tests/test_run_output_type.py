import json
import os
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

from lm_eval.api.group import Group
from lm_eval.api.task import Task
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.tasks import TaskManager


@pytest.fixture
def catalog(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"question": "Say lol", "answer": "lol"}\n')
    base = {
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(data)}},
        "test_split": "test",
        "doc_to_text": "question",
        "doc_to_target": "answer",
        "metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    metric = [{"metric": "exact_match", "aggregation": "mean"}]
    files = {
        "template": base,
        "writer.yaml": {"include": "template", "task": "writer", "tag": "writers"},
        "overridden.yaml": {
            "include": "template",
            "task": "overridden",
            "output_type": "multiple_choice",
        },
        "complete.yaml": {
            "group": "complete",
            "task": [
                "writers",
                {"task": "overridden", "output_type": "generate_until"},
            ],
            "aggregate_metric_list": metric,
        },
        "mixed.yaml": {
            "group": "mixed",
            "task": ["complete", {"group": "excluded", "task": ["bad"]}],
            "aggregate_metric_list": metric,
        },
        "root.yaml": {
            "group": "root",
            "task": ["mixed"],
            "aggregate_metric_list": metric,
        },
    }
    for name, cfg in files.items():
        (tmp_path / name).write_text(yaml.safe_dump(cfg))
    # The excluded task would fail even at import time if selection happened too late.
    (tmp_path / "bad.yaml").write_text(
        "task: bad\noutput_type: multiple_choice\ncustom_dataset: !function missing.download\n"
    )
    return tmp_path


def test_cli_partial_group_scores(catalog, tmp_path):
    output = tmp_path / "results"
    run = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "lm_eval",
            "run",
            "--model",
            "dummy",
            "--tasks",
            "root",
            "--output-type",
            "generate_until",
            "--include_path",
            str(catalog),
            "--output_path",
            str(output),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "LMEVAL_LOG_LEVEL": "INFO"},
    )
    (result_file,) = output.rglob("results*.json")
    result = json.loads(result_file.read_text())
    assert set(result["configs"]) == {"writer", "overridden"}
    assert result["config"]["output_type"] == "generate_until"
    assert result["groups"].keys() == {"complete"}
    assert result["groups"]["complete"]["exact_match,none"] == 1
    assert result["results"]["writer"]["exact_match,none"] == 1
    assert result["results"]["overridden"]["exact_match,none"] == 1
    assert "exact_match" in run.stdout
    for group in ("root", "mixed"):
        assert group not in run.stdout
        assert "exact_match,none" not in result["results"][group]
        assert f"Group {group}: selected 2/3 tasks" in run.stderr
    assert result["group_subtasks"] == {
        "root": ["mixed"],
        "mixed": ["complete"],
        "complete": ["writer", "overridden"],
    }


def test_filter_yaml_paths_and_no_matches(catalog):
    manager = TaskManager(include_path=catalog, include_defaults=False)
    loaded = manager.load(str(catalog / "mixed.yaml"), output_type="generate_until")
    assert set(loaded["tasks"]) == {"writer", "overridden"}
    with pytest.raises(
        ValueError, match="No selected tasks match output type 'loglikelihood'"
    ):
        manager.load(str(catalog / "bad.yaml"), output_type="loglikelihood")
    # Reusing the manager does not retain the previous filter.
    assert manager.load("writers")["tasks"].keys() == {"writer"}


def test_prebuilt_tasks_keep_original_group_membership():
    group = Group("original")
    for name, output_type in [
        ("choice", "multiple_choice"),
        ("score", "loglikelihood"),
    ]:
        task = Mock(spec=Task, task_name=name, OUTPUT_TYPE=output_type)
        group.add(task)
    manager = TaskManager(include_defaults=False)
    result = manager.load(group, output_type="multiple_choice")
    assert result["tasks"].keys() == {"choice"}
    assert group.child_names == ["choice", "score"]


def test_reject_invalid_config_output_type():
    with pytest.raises(ValueError, match="Unknown output_type"):
        EvaluatorConfig(tasks=["example"], output_type="greedy")._configure()


def test_python_task_uses_runtime_output_type():
    class RuntimeTask:
        def __new__(cls):
            return Mock(
                spec=Task, task_name="custom", OUTPUT_TYPE="loglikelihood_rolling"
            )

    manager = TaskManager(include_defaults=False)
    spec = {"task": "custom", "class": RuntimeTask}
    assert manager.load(spec, output_type="loglikelihood_rolling")["tasks"].keys() == {
        "custom"
    }
    with pytest.raises(ValueError, match="No selected tasks"):
        manager.load(spec, output_type="generate_until")


def test_explicit_yaml_files_skip_excluded_imports(catalog, monkeypatch):
    from functools import partial

    monkeypatch.setattr(
        "lm_eval.tasks.TaskManager", partial(TaskManager, include_defaults=False)
    )
    cfg = EvaluatorConfig(
        tasks=[str(catalog / "writer.yaml"), str(catalog / "bad.yaml")],
        output_type="generate_until",
    )
    manager = cfg.process_tasks()
    assert manager.load(cfg.tasks, output_type=cfg.output_type)["tasks"].keys() == {
        "writer"
    }
