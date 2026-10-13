"""Offline coverage for the results-directory script and its legacy JSON format."""

import copy
import json
import runpy
from pathlib import Path

import pytest

from scripts import make_table_results


SCRIPT_PATH = Path(make_table_results.__file__).resolve()


def table_rows(markdown):
    return [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in markdown.splitlines()[2:]
    ]


@pytest.mark.parametrize("with_stderr", [False, True])
@pytest.mark.parametrize(
    "task,metric,value,stderr,expected_value,expected_stderr",
    [
        ("arc_easy", "acc", 0.123456, 0.005678, 12.35, 0.57),
        ("wikitext", "ppl", 12.3456, 0.05678, 12.35, 0.06),
        ("squad2", "exact", 75.1234, 1.2345, 75.12, 1.23),
    ],
)
def test_make_table_metric_scaling(
    task, metric, value, stderr, expected_value, expected_stderr, with_stderr
):
    metrics = {metric: value}
    if with_stderr:
        metrics[f"{metric}_stderr"] = stderr
    result = {"results": {task: metrics}, "versions": {task: 3}}

    markdown = make_table_results.make_table(result)

    assert [cell.strip() for cell in markdown.splitlines()[0].split("|")[1:-1]] == [
        "Task",
        "Version",
        "Metric",
        "Value",
        "",
        "Stderr",
    ]
    (row,) = table_rows(markdown)
    assert row[:3] == [task, "3", metric]
    assert float(row[3]) == expected_value
    if with_stderr:
        assert row[4] == "±"
        assert float(row[5]) == expected_stderr
    else:
        assert row[4:] == ["", ""]


def test_make_table_sorts_tasks_preserves_metric_order_and_blanks_metadata():
    result = {
        "results": {
            "z_task": {"acc": 0.25},
            "a_task": {"z_metric_stderr": 0.01, "z_metric": 0.5, "a_metric": 0.75},
        },
        "versions": {"z_task": 2, "a_task": 1},
    }

    rows = table_rows(make_table_results.make_table(result))

    assert [row[:3] for row in rows] == [
        ["a_task", "1", "z_metric"],
        ["", "", "a_metric"],
        ["z_task", "2", "acc"],
    ]
    assert [float(row[3]) for row in rows] == [50.0, 75.0, 25.0]
    assert rows[0][4] == "±"
    assert float(rows[0][5]) == 1.0
    assert rows[1][4:] == rows[2][4:] == ["", ""]


def test_make_table_keeps_squad_scale_after_the_first_metric():
    result = {
        "results": {"squad2": {"exact": 75.25, "f1": 82.5, "f1_stderr": 1.5}},
        "versions": {"squad2": 2},
    }

    rows = table_rows(make_table_results.make_table(result))

    assert [row[:3] for row in rows] == [
        ["squad2", "2", "exact"],
        ["", "", "f1"],
    ]
    assert [float(row[3]) for row in rows] == [75.25, 82.5]
    assert rows[1][4] == "±"
    assert float(rows[1][5]) == 1.5


@pytest.mark.parametrize("metrics", [{}, {"acc_stderr": 0.1}])
def test_make_table_does_not_emit_stderr_only_rows(metrics):
    result = {"results": {"task": metrics}, "versions": {"task": 1}}

    assert table_rows(make_table_results.make_table(result)) == []


def test_make_table_empty_results():
    assert (
        table_rows(make_table_results.make_table({"results": {}, "versions": {}})) == []
    )


def test_make_table_zero_scores_and_errors_are_not_omitted():
    result = {
        "results": {"task": {"acc": 0.0, "acc_stderr": 0.0}},
        "versions": {"task": 0},
    }

    (row,) = table_rows(make_table_results.make_table(result))

    assert row[:3] == ["task", "0", "acc"]
    assert float(row[3]) == 0.0
    assert row[4] == "±"
    assert float(row[5]) == 0.0


def test_make_table_does_not_mutate_input():
    result = {
        "results": {"task": {"acc": 0.5, "acc_stderr": 0.01, "ppl": 12.5}},
        "versions": {"task": 1},
    }
    original = copy.deepcopy(result)

    first = make_table_results.make_table(result)
    second = make_table_results.make_table(result)

    assert result == original
    assert first == second


def run_script(working_directory, monkeypatch):
    monkeypatch.chdir(working_directory)
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


def test_main_writes_sorted_json_tables_in_nested_directories(tmp_path, monkeypatch):
    working_directory = tmp_path / "scripts"
    working_directory.mkdir()
    model_directory = tmp_path / "results" / "model"
    nested_directory = model_directory / "nested"
    nested_directory.mkdir(parents=True)
    (model_directory / "README.md").write_text("stale content", encoding="utf-8")
    (model_directory / "notes.txt").write_text("not JSON", encoding="utf-8")
    result = {"results": {"task": {"acc": 0.5}}, "versions": {"task": 1}}
    for name in ("z_results.json", "a_results.json"):
        (model_directory / name).write_text(json.dumps(result), encoding="utf-8")
    (nested_directory / "results.json").write_text(json.dumps(result), encoding="utf-8")

    run_script(working_directory, monkeypatch)

    markdown = (model_directory / "README.md").read_text(encoding="utf-8")
    assert markdown.startswith("# model \n\n## a_results.json \n")
    assert markdown.index("## a_results.json") < markdown.index("## z_results.json")
    assert markdown.count("|Task|") == 2
    rows = [
        line.split("|") for line in markdown.splitlines() if line.startswith("|task")
    ]
    assert [float(row[4]) for row in rows] == [50.0, 50.0]
    assert "stale content" not in markdown
    assert "notes.txt" not in markdown
    assert (model_directory / "notes.txt").read_text(encoding="utf-8") == "not JSON"
    assert (nested_directory / "README.md").read_text(encoding="utf-8") == (
        "# nested \n\n## results.json \n"
        + make_table_results.make_table(result)
        + " \n"
    )


@pytest.mark.parametrize("create_results_directory", [False, True])
def test_main_skips_missing_or_empty_result_directories(
    tmp_path, monkeypatch, create_results_directory
):
    working_directory = tmp_path / "scripts"
    working_directory.mkdir()
    results_directory = tmp_path / "results"
    if create_results_directory:
        (results_directory / "empty_model").mkdir(parents=True)

    run_script(working_directory, monkeypatch)

    assert not list(results_directory.rglob("README.md"))


def test_main_reports_invalid_json(tmp_path, monkeypatch):
    working_directory = tmp_path / "scripts"
    working_directory.mkdir()
    results_directory = tmp_path / "results"
    results_directory.mkdir()
    (results_directory / "bad.json").write_text("not JSON", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        run_script(working_directory, monkeypatch)
