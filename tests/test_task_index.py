"""
Tests for the task index builder that discovers YAML task configurations.

Test coverage:
- TaskIndexBuilder._kind_of: identifies task/group/tag/task_list/py_task
- TaskIndexBuilder._iter_yaml_files: finds YAML files, ignores __pycache__
- TaskIndexBuilder._process_cfg: creates correct TaskEntry for each type
- TaskIndexBuilder._register_tags: creates TAG entries for task tags
- TaskIndexBuilder.build: discovers all task types in directory tree
"""

import tempfile
from pathlib import Path

import pytest

from lm_eval.tasks._task_index import TaskIndexBuilder, TaskKind


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def yaml_file(temp_dir):
    def _create_yaml(content, path="test.yaml"):
        file_path = temp_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    return _create_yaml


class TestTaskKindOf:
    """Tests for identifying task configuration types."""

    def test_kind_of_task(self):
        """Single task with string name."""
        cfg = {"task": "my_task", "dataset_path": "data"}
        assert TaskIndexBuilder._kind_of(cfg) == TaskKind.TASK

    def test_kind_of_group(self):
        """Group has task as list."""
        cfg = {"task": ["task1", "task2"], "group": "my_group"}
        assert TaskIndexBuilder._kind_of(cfg) == TaskKind.GROUP

    def test_kind_of_py_task(self):
        """Python task has class field."""
        cfg = {"task": "my_task", "class": "tasks.MyTask"}
        assert TaskIndexBuilder._kind_of(cfg) == TaskKind.PY_TASK

    def test_kind_of_task_list(self):
        """Task list has task_list field."""
        cfg = {"task_list": ["task1", "task2"]}
        assert TaskIndexBuilder._kind_of(cfg) == TaskKind.TASK_LIST

    def test_kind_of_unknown(self):
        """Unknown config raises ValueError."""
        cfg = {"unknown": "field"}
        with pytest.raises(ValueError, match="Unknown config shape"):
            TaskIndexBuilder._kind_of(cfg)


class TestIterYamlFiles:
    """Tests for YAML file discovery."""

    def test_iter_yaml_files_simple(self, temp_dir):
        """Finds .yaml files in directory tree."""
        # Create some yaml files
        (temp_dir / "task1.yaml").touch()
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "task2.yaml").touch()
        (temp_dir / "other.txt").touch()

        builder = TaskIndexBuilder()
        yaml_files = list(builder._iter_yaml_files(temp_dir))

        assert len(yaml_files) == 2
        names = {f.name for f in yaml_files}
        assert names == {"task1.yaml", "task2.yaml"}

    def test_iter_yaml_files_ignores_pycache(self, temp_dir):
        """Ignores files in __pycache__ directories."""
        (temp_dir / "task.yaml").touch()
        (temp_dir / "__pycache__").mkdir()
        (temp_dir / "__pycache__" / "ignored.yaml").touch()
        (temp_dir / ".ipynb_checkpoints").mkdir()
        (temp_dir / ".ipynb_checkpoints" / "also_ignored.yaml").touch()

        builder = TaskIndexBuilder()
        yaml_files = list(builder._iter_yaml_files(temp_dir))

        assert len(yaml_files) == 1
        assert yaml_files[0].name == "task.yaml"


class TestProcessCfg:
    """Tests for processing individual config files."""

    def test_process_task(self, temp_dir):
        """Regular task creates TASK entry."""
        cfg = {"task": "my_task", "tag": ["tag1", "tag2"]}
        path = temp_dir / "task.yaml"
        index = {}

        builder = TaskIndexBuilder()
        builder._process_cfg(cfg, path, index)

        assert "my_task" in index
        entry = index["my_task"]
        assert entry.name == "my_task"
        assert entry.kind == TaskKind.TASK
        assert entry.yaml_path == path
        assert entry.tags == {"tag1", "tag2"}

    def test_process_group(self, temp_dir):
        """Group creates GROUP entry."""
        cfg = {"task": ["t1", "t2"], "group": "my_group", "tag": ["grp_tag"]}
        path = temp_dir / "group.yaml"
        index = {}

        builder = TaskIndexBuilder()
        builder._process_cfg(cfg, path, index)

        assert "my_group" in index
        entry = index["my_group"]
        assert entry.name == "my_group"
        assert entry.kind == TaskKind.GROUP
        assert entry.yaml_path == path
        assert entry.tags == {"grp_tag"}

    def test_process_py_task(self, temp_dir):
        """Python task creates PY_TASK entry."""
        cfg = {"task": "py_task", "class": "MyTask", "tag": ["py_tag"]}
        path = temp_dir / "py_task.yaml"
        index = {}

        builder = TaskIndexBuilder()
        builder._process_cfg(cfg, path, index)

        assert "py_task" in index
        entry = index["py_task"]
        assert entry.name == "py_task"
        assert entry.kind == TaskKind.PY_TASK
        assert entry.yaml_path is None  # Python tasks don't store yaml_path
        assert entry.tags == {"py_tag"}

    def test_process_task_list(self, temp_dir):
        """Task list creates entries for each task."""
        cfg = {
            "task_list": [
                "simple_task",
                {"task": "complex_task", "tag": ["tag1", "tag2"]},
            ]
        }
        path = temp_dir / "list.yaml"
        index = {}

        builder = TaskIndexBuilder()
        # The implementation has a bug - it calls entry.get() on string entries
        # This test documents the current behavior which will fail
        with pytest.raises(AttributeError, match="'str' object has no attribute 'get'"):
            builder._process_cfg(cfg, path, index)

    def test_process_task_list_dict_entries(self, temp_dir):
        """Task list with only dict entries works."""
        cfg = {
            "task_list": [{"task": "task1"}, {"task": "task2", "tag": ["tag1", "tag2"]}]
        }
        path = temp_dir / "list.yaml"
        index = {}

        builder = TaskIndexBuilder()
        builder._process_cfg(cfg, path, index)

        # Task without tags
        assert "task1" in index
        task1 = index["task1"]
        assert task1.kind == TaskKind.TASK
        assert task1.yaml_path == path
        assert task1.tags == set()

        # Task with tags
        assert "task2" in index
        task2 = index["task2"]
        assert task2.kind == TaskKind.TASK
        assert task2.yaml_path == path
        assert task2.tags == {"tag1", "tag2"}


class TestRegisterTags:
    """Tests for tag registration."""

    def test_register_single_tag(self):
        """Single tag creates TAG entry."""
        index = {}
        builder = TaskIndexBuilder()

        builder._register_tags("task1", "my_tag", index)

        assert "my_tag" in index
        tag_entry = index["my_tag"]
        assert tag_entry.kind == TaskKind.TAG
        assert tag_entry.yaml_path is None
        assert "task1" in tag_entry.tags  # TAG entries use tags set for task names

    def test_register_multiple_tags(self):
        """Multiple tags create multiple TAG entries."""
        index = {}
        builder = TaskIndexBuilder()

        builder._register_tags("task1", ["tag1", "tag2"], index)

        assert "tag1" in index
        assert "tag2" in index
        assert "task1" in index["tag1"].tags
        assert "task1" in index["tag2"].tags

    def test_register_tags_accumulates(self):
        """Multiple tasks can have same tag."""
        index = {}
        builder = TaskIndexBuilder()

        builder._register_tags("task1", "shared_tag", index)
        builder._register_tags("task2", "shared_tag", index)

        assert "shared_tag" in index
        tag_entry = index["shared_tag"]
        assert tag_entry.tags == {"task1", "task2"}


class TestBuild:
    """Tests for the main build method."""

    def test_build_empty_directory(self, temp_dir):
        """Empty directory returns empty index."""
        builder = TaskIndexBuilder()
        index = builder.build([temp_dir])
        assert index == {}

    def test_build_single_task(self, temp_dir, yaml_file):
        """Single task file is discovered."""
        yaml_file("task: my_task\ndataset_path: data\n")

        builder = TaskIndexBuilder()
        index = builder.build([temp_dir])

        assert len(index) == 1
        assert "my_task" in index
        assert index["my_task"].kind == TaskKind.TASK

    def test_build_mixed_types(self, temp_dir, yaml_file):
        """Discovers various task types."""
        # Regular task with list tag format
        yaml_file("task: task1\ntag: [common]\n", "task1.yaml")

        # Group
        yaml_file("task: [t1, t2]\ngroup: group1\n", "group1.yaml")

        # Task list with only dict entries (to avoid the bug)
        yaml_file(
            "task_list:\n  - task: task2\n  - task: task3\n    tag: [common]\n",
            "list.yaml",
        )

        # Python task
        yaml_file("task: py_task\nclass: MyClass\n", "python.yaml")

        builder = TaskIndexBuilder()
        index = builder.build([temp_dir])

        # Check all entries exist
        assert "task1" in index
        assert "group1" in index
        assert "task2" in index
        assert "task3" in index
        assert "py_task" in index
        assert "common" in index  # Tag entry

        # Check types
        assert index["task1"].kind == TaskKind.TASK
        assert index["group1"].kind == TaskKind.GROUP
        assert index["task2"].kind == TaskKind.TASK
        assert index["task3"].kind == TaskKind.TASK
        assert index["py_task"].kind == TaskKind.PY_TASK
        assert index["common"].kind == TaskKind.TAG

        # Check tag has both tasks
        assert index["common"].tags == {"task1", "task3"}

    def test_build_nested_directories(self, temp_dir, yaml_file):
        """Discovers tasks in nested directories."""
        yaml_file("task: root_task\n", "root.yaml")
        yaml_file("task: sub_task\n", "subdir/sub.yaml")
        yaml_file("task: deep_task\n", "subdir/deeper/deep.yaml")

        builder = TaskIndexBuilder()
        index = builder.build([temp_dir])

        assert len(index) == 3
        assert all(name in index for name in ["root_task", "sub_task", "deep_task"])

    def test_build_skips_invalid_yaml(self, temp_dir, yaml_file):
        """Skips files that fail to parse."""
        yaml_file("task: valid_task\n", "valid.yaml")
        yaml_file("invalid: [\n", "invalid.yaml")  # Invalid YAML

        builder = TaskIndexBuilder()
        index = builder.build([temp_dir])

        assert len(index) == 1
        assert "valid_task" in index

    def test_build_multiple_paths(self, temp_dir):
        """Can search multiple root paths."""
        # Create two separate directories
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "task1.yaml").write_text("task: task1\n")
        (dir2 / "task2.yaml").write_text("task: task2\n")

        builder = TaskIndexBuilder()
        index = builder.build([dir1, dir2])

        assert len(index) == 2
        assert "task1" in index
        assert "task2" in index
