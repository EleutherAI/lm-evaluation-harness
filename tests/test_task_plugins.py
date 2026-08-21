"""Tests for task-directory plugin discovery via entry points."""

from pathlib import Path

import lm_eval.tasks.manager as manager_mod
from lm_eval.tasks.manager import TaskManager, discover_plugin_task_paths


_TASK_YAML = """\
task: {name}
dataset_path: EleutherAI/lambada_openai
dataset_name: default
output_type: loglikelihood
test_split: test
doc_to_text: "{{{{text}}}}"
doc_to_target: "{{{{text}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: {version}
"""


def _make_task_package(tmp_path, pkg_name, task_name, version=1.0):
    """Create an importable package on sys.path with one task YAML inside it."""
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / f"{task_name}.yaml").write_text(
        _TASK_YAML.format(name=task_name, version=version)
    )
    return pkg_dir


def _patch_task_entry_points(monkeypatch, eps):
    def fake_entry_points(*, group):
        return eps if group == manager_mod.TASKS_ENTRY_POINT_GROUP else []

    monkeypatch.setattr(manager_mod.md, "entry_points", fake_entry_points)


def _ep(name, value):
    return manager_mod.md.EntryPoint(
        name=name, value=value, group=manager_mod.TASKS_ENTRY_POINT_GROUP
    )


def test_resolve_task_dir_for_package(tmp_path, monkeypatch):
    _make_task_package(tmp_path, "plug_pkg_a", "plug_task_a")
    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = manager_mod._resolve_task_dir("plug_pkg_a")

    assert resolved == (tmp_path / "plug_pkg_a").resolve()


def test_discover_plugin_task_paths(tmp_path, monkeypatch):
    _make_task_package(tmp_path, "plug_pkg_b", "plug_task_b")
    monkeypatch.syspath_prepend(str(tmp_path))
    _patch_task_entry_points(monkeypatch, [_ep("plug_b", "plug_pkg_b")])

    paths = discover_plugin_task_paths()

    assert (tmp_path / "plug_pkg_b").resolve() in paths


def test_task_manager_includes_plugin_task(tmp_path, monkeypatch):
    _make_task_package(tmp_path, "plug_pkg_c", "plug_task_c")
    monkeypatch.syspath_prepend(str(tmp_path))
    _patch_task_entry_points(monkeypatch, [_ep("plug_c", "plug_pkg_c")])

    # include_defaults=False keeps this fast and isolated to the plugin path.
    tm = TaskManager(include_defaults=False)

    assert "plug_task_c" in tm.all_tasks


def test_include_path_overrides_plugin_task(tmp_path, monkeypatch):
    """A user --include_path task shadows a plugin task of the same name."""
    _make_task_package(tmp_path, "plug_pkg_d", "shared_task", version=1.0)
    monkeypatch.syspath_prepend(str(tmp_path))
    _patch_task_entry_points(monkeypatch, [_ep("plug_d", "plug_pkg_d")])

    override_dir = tmp_path / "override"
    override_dir.mkdir()
    (override_dir / "shared_task.yaml").write_text(
        _TASK_YAML.format(name="shared_task", version=2.0)
    )

    tm = TaskManager(include_path=str(override_dir), include_defaults=False)

    entry = tm.task_index["shared_task"]
    # The include_path copy (version 2.0) must win over the plugin copy.
    assert Path(entry.yaml_path).parent == override_dir.resolve() or str(
        override_dir
    ) in str(entry.yaml_path)


def test_broken_task_entry_point_is_tolerated(monkeypatch):
    _patch_task_entry_points(
        monkeypatch, [_ep("broken", "nonexistent_module_xyz.tasks")]
    )

    # Must not raise - a bad plugin is logged and skipped.
    paths = discover_plugin_task_paths()

    assert all("nonexistent_module_xyz" not in str(p) for p in paths)
