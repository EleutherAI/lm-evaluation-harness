import logging
import tempfile
from pathlib import Path

import pytest

from lm_eval.tasks import TaskManager
from lm_eval.tasks._config_loader import load_yaml
from lm_eval.tasks.index import Entry, Kind, TaskIndex


# =============================================================================
# Existing fixtures and tests
# =============================================================================


@pytest.fixture(scope="module")
def custom_task_name():
    return "zzz_my_python_task"


@pytest.fixture(scope="module")
def custom_task_tag():
    return "zzz-tag"


@pytest.fixture(scope="module")
def task_yaml(pytestconfig, custom_task_name, custom_task_tag):
    yield f"""include: {pytestconfig.rootpath}/lm_eval/tasks/arc/arc_easy.yaml
task: {custom_task_name}
class: !function {custom_task_name}.MockPythonTask
tag:
  - {custom_task_tag}
"""


@pytest.fixture(scope="module")
def task_code():
    return """
from lm_eval.tasks import ConfigurableTask

class MockPythonTask(ConfigurableTask):

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        config.pop("class")
        super().__init__(data_dir, cache_dir, download_mode, config)
"""


@pytest.fixture(scope="module")
def custom_task_files_dir(task_yaml, task_code, custom_task_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = Path(temp_dir) / f"{custom_task_name}.yaml"
        with open(yaml_path, "w") as f:
            f.write(task_yaml)
        pysource_path = Path(temp_dir) / f"{custom_task_name}.py"
        with open(pysource_path, "w") as f:
            f.write(task_code)
        yield temp_dir


def test_python_task_inclusion(
    custom_task_files_dir: Path, custom_task_name: str, custom_task_tag: str
):
    task_manager = TaskManager(
        verbosity="INFO", include_path=str(custom_task_files_dir)
    )
    # check if python tasks enters the global task_index
    assert custom_task_name in task_manager.task_index
    # check if subtask is present
    assert custom_task_name in task_manager.all_subtasks
    # check if tag is present
    assert custom_task_tag in task_manager.all_tags
    # check if it can be loaded by tag (custom_task_tag)
    assert custom_task_name in task_manager.load_task_or_group(custom_task_tag)


# =============================================================================
# Config Loader Tests
# =============================================================================


class TestConfigLoader:
    def test_load_simple_yaml(self, tmp_path):
        """Load a basic YAML without includes or functions"""
        content = """
task: simple_test
dataset_path: test_dataset
output_type: generate_until
"""
        yaml_path = tmp_path / "simple.yaml"
        yaml_path.write_text(content)

        cfg = load_yaml(yaml_path)

        assert cfg["task"] == "simple_test"
        assert cfg["dataset_path"] == "test_dataset"
        assert cfg["output_type"] == "generate_until"

    def test_load_yaml_with_include(self, tmp_path):
        """Load YAML that includes another file"""
        base_content = """
dataset_path: base_dataset
output_type: multiple_choice
num_fewshot: 5
"""
        child_content = """
include: base.yaml
task: child_task
num_fewshot: 10
"""
        (tmp_path / "base.yaml").write_text(base_content)
        (tmp_path / "child.yaml").write_text(child_content)

        cfg = load_yaml(tmp_path / "child.yaml")

        # Child overrides base
        assert cfg["task"] == "child_task"
        assert cfg["num_fewshot"] == 10
        # Inherited from base
        assert cfg["dataset_path"] == "base_dataset"
        assert cfg["output_type"] == "multiple_choice"

    def test_load_yaml_with_function_tag_resolved(self, tmp_path):
        """Load YAML with !function tag, resolve_func=True"""
        utils_content = """
def my_processor(doc):
    return doc
"""
        yaml_content = """
task: func_test
process_docs: !function utils.my_processor
"""
        (tmp_path / "utils.py").write_text(utils_content)
        (tmp_path / "test.yaml").write_text(yaml_content)

        cfg = load_yaml(tmp_path / "test.yaml", resolve_func=True)

        assert cfg["task"] == "func_test"
        assert callable(cfg["process_docs"])

    def test_load_yaml_without_function_resolution(self, tmp_path):
        """Load YAML with !function tag, resolve_func=False (returns path string)"""
        yaml_content = """
task: func_test
process_docs: !function utils.my_processor
"""
        (tmp_path / "test.yaml").write_text(yaml_content)

        cfg = load_yaml(tmp_path / "test.yaml", resolve_func=False)

        assert cfg["task"] == "func_test"
        # When resolve_func=False, returns path string
        assert isinstance(cfg["process_docs"], str)
        assert "utils.my_processor" in cfg["process_docs"]

    def test_load_yaml_recursive_includes(self, tmp_path):
        """Load YAML with nested includes"""
        grandparent = """
output_type: generate_until
metric_list:
  - metric: exact_match
"""
        parent = """
include: grandparent.yaml
dataset_path: parent_dataset
"""
        child = """
include: parent.yaml
task: nested_task
"""
        (tmp_path / "grandparent.yaml").write_text(grandparent)
        (tmp_path / "parent.yaml").write_text(parent)
        (tmp_path / "child.yaml").write_text(child)

        cfg = load_yaml(tmp_path / "child.yaml")

        assert cfg["task"] == "nested_task"
        assert cfg["dataset_path"] == "parent_dataset"
        assert cfg["output_type"] == "generate_until"

    def test_load_yaml_cycle_detection(self, tmp_path):
        """Detect include cycles"""
        a_content = """
include: b.yaml
task: a
"""
        b_content = """
include: a.yaml
task: b
"""
        (tmp_path / "a.yaml").write_text(a_content)
        (tmp_path / "b.yaml").write_text(b_content)

        with pytest.raises(ValueError, match="Include cycle"):
            load_yaml(tmp_path / "a.yaml")


# =============================================================================
# TaskIndex Tests
# =============================================================================


class TestKind:
    def test_kind_enum_values(self):
        """Verify Kind enum has expected values"""
        assert Kind.TASK is not None
        assert Kind.PY_TASK is not None
        assert Kind.GROUP is not None
        assert Kind.TAG is not None
        assert Kind.TASK_LIST is not None


class TestEntry:
    def test_entry_dataclass_fields(self):
        """Verify Entry has expected fields"""
        entry = Entry(
            name="test",
            kind=Kind.TASK,
            yaml_path=Path("/test.yaml"),
            cfg={"task": "test"},
            tags={"tag1"},
        )
        assert entry.name == "test"
        assert entry.kind == Kind.TASK
        assert entry.yaml_path == Path("/test.yaml")
        assert entry.cfg == {"task": "test"}
        assert entry.tags == {"tag1"}


class TestTaskIndex:
    def test_build_from_directory(self, tmp_path):
        """Build index from a directory with YAML files"""
        task_content = """
task: test_task
dataset_path: test
output_type: generate_until
"""
        (tmp_path / "test_task.yaml").write_text(task_content)

        index = TaskIndex()
        result = index.build([tmp_path])

        assert "test_task" in result
        assert result["test_task"].kind == Kind.TASK

    def test_deterministic_traversal(self, tmp_path):
        """Verify files are processed in sorted order"""
        # Create files that would be in different order without sorting
        (tmp_path / "z_task.yaml").write_text("task: z_task\ndataset_path: z")
        (tmp_path / "a_task.yaml").write_text("task: a_task\ndataset_path: a")
        (tmp_path / "m_task.yaml").write_text("task: m_task\ndataset_path: m")

        index = TaskIndex()
        result = index.build([tmp_path])

        # All tasks should be indexed
        assert "a_task" in result
        assert "m_task" in result
        assert "z_task" in result

    def test_duplicate_task_detection(self, tmp_path, caplog):
        """Verify warning logged for duplicate task names"""
        # Create subdirectories with duplicate task names
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "task.yaml").write_text("task: duplicate_task\ndataset_path: a")
        (dir2 / "task.yaml").write_text("task: duplicate_task\ndataset_path: b")

        index = TaskIndex()
        with caplog.at_level(logging.WARNING):
            result = index.build([tmp_path])

        # Only one should be registered
        assert "duplicate_task" in result
        # Warning should be logged
        assert "Duplicate task name" in caplog.text

    def test_duplicate_group_detection(self, tmp_path, caplog):
        """Verify warning logged for duplicate group names"""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        group_content = """
group: duplicate_group
task:
  - task1
"""
        (dir1 / "group.yaml").write_text(group_content)
        (dir2 / "group.yaml").write_text(group_content)

        # Also need task1 to exist
        (tmp_path / "task1.yaml").write_text("task: task1\ndataset_path: t")

        index = TaskIndex()
        with caplog.at_level(logging.WARNING):
            result = index.build([tmp_path])

        assert "duplicate_group" in result
        assert "Duplicate group name" in caplog.text

    def test_kind_detection_task(self):
        """Config with 'task' key (string) detected as TASK"""
        cfg = {"task": "my_task", "dataset_path": "test"}
        kind = TaskIndex._kind_of(cfg)
        assert kind == Kind.TASK

    def test_kind_detection_group(self):
        """Config with 'group' key detected as GROUP"""
        cfg = {"group": "my_group", "task": ["task1", "task2"]}
        kind = TaskIndex._kind_of(cfg)
        assert kind == Kind.GROUP

    def test_kind_detection_py_task(self):
        """Config with 'class' key detected as PY_TASK"""
        cfg = {"task": "my_task", "class": "SomeClass"}
        kind = TaskIndex._kind_of(cfg)
        assert kind == Kind.PY_TASK

    def test_kind_detection_task_list(self):
        """Config with 'task_list' key detected as TASK_LIST"""
        cfg = {"task_list": [{"task": "task1"}, {"task": "task2"}]}
        kind = TaskIndex._kind_of(cfg)
        assert kind == Kind.TASK_LIST

    def test_tag_registration(self, tmp_path):
        """Tags from tasks are registered in index"""
        task_content = """
task: tagged_task
dataset_path: test
tag: my_custom_tag
"""
        (tmp_path / "task.yaml").write_text(task_content)

        index = TaskIndex()
        result = index.build([tmp_path])

        assert "tagged_task" in result
        assert "my_custom_tag" in result
        assert result["my_custom_tag"].kind == Kind.TAG
        assert "tagged_task" in result["my_custom_tag"].tags

    def test_ignore_pycache(self, tmp_path):
        """Files in __pycache__ are ignored"""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "task.yaml").write_text("task: should_ignore\ndataset_path: t")

        index = TaskIndex()
        result = index.build([tmp_path])

        assert "should_ignore" not in result


# =============================================================================
# TaskManager Integration Tests
# =============================================================================


# Module-level fixture to avoid re-creating TaskManager for each test
@pytest.fixture(scope="module")
def shared_task_manager():
    """Create a TaskManager with default tasks (shared across module)"""
    return TaskManager()


class TestTaskManagerIntegration:
    def test_initialization(self, shared_task_manager):
        """TaskManager initializes with default tasks"""
        assert len(shared_task_manager.all_tasks) > 0

    def test_all_tasks_sorted(self, shared_task_manager):
        """all_tasks returns sorted list"""
        tasks = shared_task_manager.all_tasks
        assert tasks == sorted(tasks)

    def test_all_groups_property(self, shared_task_manager):
        """all_groups returns only groups"""
        groups = shared_task_manager.all_groups
        assert len(groups) > 0
        for g in groups[:5]:  # Check first 5
            assert shared_task_manager._name_is_group(g)

    def test_all_subtasks_property(self, shared_task_manager):
        """all_subtasks returns TASK and PY_TASK kinds"""
        subtasks = shared_task_manager.all_subtasks
        assert len(subtasks) > 0
        for t in subtasks[:5]:  # Check first 5
            entry = shared_task_manager.task_index[t]
            assert entry.kind in (Kind.TASK, Kind.PY_TASK)

    def test_all_tags_property(self, shared_task_manager):
        """all_tags returns only tags"""
        tags = shared_task_manager.all_tags
        assert len(tags) > 0
        for t in tags[:5]:  # Check first 5
            assert shared_task_manager._name_is_tag(t)

    def test_load_task_by_name(self, shared_task_manager):
        """Load a single task by name"""
        result = shared_task_manager.load_task_or_group(["arc_easy"])
        assert "arc_easy" in result

    def test_load_group_by_name(self, shared_task_manager):
        """Load a group and get nested structure"""
        result = shared_task_manager.load_task_or_group(["ai2_arc"])
        # ai2_arc is a tag that contains arc_easy and arc_challenge
        assert "arc_easy" in result or "arc_challenge" in result

    def test_load_tag_by_name(self, shared_task_manager):
        """Load all tasks in a tag"""
        result = shared_task_manager.load_task_or_group(["ai2_arc"])
        # Should load both arc_easy and arc_challenge
        assert "arc_easy" in result
        assert "arc_challenge" in result

    def test_include_path(self):
        """Custom include_path adds tasks to index using tests/test_configs/"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)
        # simple_task is defined in test_configs/simple_task.yaml
        assert "simple_task" in tm.all_tasks

    def test_include_defaults_false(self):
        """include_defaults=False excludes built-in tasks"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)
        # Should have tasks from test_configs
        assert "simple_task" in tm.all_tasks
        # Built-in tasks like arc_easy should not be present
        assert "arc_easy" not in tm.all_tasks

    def test_include_resolution(self):
        """Test that includes are properly resolved"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)
        # include_task_fs5 includes include_base which has the actual task config
        assert "include_task_fs5" in tm.all_tasks

    def test_include_inheritance_override(self):
        """Test that child config overrides parent values from include"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)

        # Load the task to get full resolved config
        result = tm.load_task_or_group(["include_task_fs5"])
        task_obj = result["include_task_fs5"]

        # include_base has num_fewshot=0, include_task_fs5 overrides to 5
        assert task_obj.config.num_fewshot == 5

        # include_base has dataset_path=json (inherited)
        assert task_obj.config.dataset_path == "json"

    def test_include_custom_metrics(self):
        """Test that include_task_fs5 has custom metrics (acc + acc_norm)"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)

        result = tm.load_task_or_group(["include_task_fs5"])
        task_obj = result["include_task_fs5"]

        # include_task_fs5 defines both acc and acc_norm metrics
        metric_names = [m["metric"] for m in task_obj.config.metric_list]
        assert "acc" in metric_names
        assert "acc_norm" in metric_names

    def test_group_loading(self):
        """Test that groups are indexed from test_configs"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)
        # group.yaml defines a group called 'test_group'
        assert "test_group" in tm.all_groups

    def test_include_group(self):
        """Test group with tasks sharing same base config via includes"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)
        # include_group.yaml: group with include_task_fs0, fs1, fs5
        assert "include_group" in tm.all_groups
        # The subtasks should also be indexed
        assert "include_task_fs0" in tm.all_tasks
        assert "include_task_fs1" in tm.all_tasks
        assert "include_task_fs5" in tm.all_tasks

    def test_task_list_loading(self):
        """Test task_list feature with shared config and task-specific overrides"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)
        # task_list.yaml defines tasks via task_list key
        assert "task_list_fs0" in tm.all_tasks
        assert "task_list_fs1" in tm.all_tasks
        assert "task_list_fs3" in tm.all_tasks

    def test_task_list_overrides(self):
        """Test task_list task-specific overrides are applied"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)

        # Load tasks and verify their num_fewshot values
        result = tm.load_task_or_group(
            ["task_list_fs0", "task_list_fs1", "task_list_fs3"]
        )

        assert result["task_list_fs0"].config.num_fewshot == 0
        assert result["task_list_fs1"].config.num_fewshot == 1
        assert result["task_list_fs3"].config.num_fewshot == 3

        # task_list_fs3 has custom metrics (acc + acc_norm)
        metric_names = [m["metric"] for m in result["task_list_fs3"].config.metric_list]
        assert "acc" in metric_names
        assert "acc_norm" in metric_names

    def test_match_tasks_glob(self, shared_task_manager):
        """match_tasks handles glob patterns"""
        matches = shared_task_manager.match_tasks(["arc_*"])
        assert "arc_easy" in matches
        assert "arc_challenge" in matches

    def test_name_is_registered(self, shared_task_manager):
        """_name_is_registered checks if name exists"""
        assert shared_task_manager._name_is_registered("arc_easy")
        assert not shared_task_manager._name_is_registered("nonexistent_task_xyz")

    def test_name_is_task(self, shared_task_manager):
        """_name_is_task returns True for tasks"""
        assert shared_task_manager._name_is_task("arc_easy")
        assert not shared_task_manager._name_is_task("ai2_arc")  # This is a tag

    def test_name_is_tag(self, shared_task_manager):
        """_name_is_tag returns True for tags"""
        assert shared_task_manager._name_is_tag("ai2_arc")
        assert not shared_task_manager._name_is_tag("arc_easy")  # This is a task

    def test_include_path_precedence(self):
        """Test that user-specified include paths take precedence over default paths when tasks have the same name."""
        with tempfile.TemporaryDirectory() as custom_dir:
            # Create a custom arc_easy.yaml that has a different metric
            custom_task_content = """task: arc_easy
dataset_path: allenai/ai2_arc
dataset_name: ARC-Easy
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Custom Question: {{question}}\\nAnswer:"
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{choices.text}}"
metric_list:
  - metric: f1
    aggregation: mean
    higher_is_better: true
metadata:
  version: 2.0
  custom: true
"""
            # Write the custom task file
            custom_task_path = Path(custom_dir) / "arc_easy.yaml"
            custom_task_path.write_text(custom_task_content)

            # Test 1: User path should override default when include_defaults=True
            task_manager = TaskManager(include_defaults=True, include_path=custom_dir)

            # Load the task
            task_dict = task_manager.load_task_or_group(["arc_easy"])
            arc_easy_task = task_dict["arc_easy"]

            # Check that the custom version was loaded (has f1 metric and custom doc_to_text)
            assert any(
                metric["metric"] == "f1"
                for metric in arc_easy_task.config["metric_list"]
            ), "Custom task should have f1 metric"
            assert "Custom Question:" in arc_easy_task.config["doc_to_text"], (
                "Custom task should have custom doc_to_text"
            )
            assert arc_easy_task.config["metadata"]["version"] == 2.0, (
                "Custom task should have version 2.0"
            )

            # Test 2: Verify default is used when no custom path is provided
            default_task_manager = TaskManager(include_defaults=True)
            default_task_dict = default_task_manager.load_task_or_group(["arc_easy"])
            default_arc_easy = default_task_dict["arc_easy"]

            # Default should not have f1 metric or custom text
            assert not any(
                metric["metric"] == "f1"
                for metric in default_arc_easy.config.get("metric_list", [])
            ), "Default task should not have f1 metric"
            assert "Custom Question:" not in default_arc_easy.config["doc_to_text"], (
                "Default task should not have custom doc_to_text"
            )

    def test_include_defaults_false_with_custom_path(self):
        """Test that when include_defaults=False, only custom tasks are available."""
        with tempfile.TemporaryDirectory() as custom_dir:
            # Create a custom task using a real dataset
            custom_task_content = """task: custom_arc_task
dataset_path: allenai/ai2_arc
dataset_name: ARC-Challenge
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Q: {{question}}\nA:"
doc_to_target: "{{choices.label.index(answerKey)}}"
doc_to_choice: "{{choices.text}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
  custom: true
"""
            # Write the custom task file
            custom_task_path = Path(custom_dir) / "custom_arc_task.yaml"
            custom_task_path.write_text(custom_task_content)

            # Initialize with include_defaults=False
            task_manager = TaskManager(include_defaults=False, include_path=custom_dir)

            # Custom task should be available
            assert "custom_arc_task" in task_manager.all_tasks, (
                "Custom task should be available when include_defaults=False"
            )

            # Default tasks should NOT be available
            assert "arc_easy" not in task_manager.all_tasks, (
                "Default arc_easy should not be available when include_defaults=False"
            )
            assert "arc_challenge" not in task_manager.all_tasks, (
                "Default arc_challenge should not be available when include_defaults=False"
            )

            # Check that only our custom task is present
            assert len(task_manager.all_tasks) == 1, (
                f"Should only have 1 task, but found {len(task_manager.all_tasks)}"
            )

            # Check task metadata using Entry object API
            entry = task_manager.task_index["custom_arc_task"]
            assert entry.kind == Kind.TASK
            assert custom_dir in str(entry.yaml_path)

    def test_include_defaults_true_with_new_tasks(self, shared_task_manager):
        """Test that new tasks from include_path are added alongside default tasks."""
        with tempfile.TemporaryDirectory() as custom_dir:
            # Create a completely new task (not overriding any default)
            new_task_content = """task: arc_custom_generation
dataset_path: allenai/ai2_arc
dataset_name: ARC-Easy
output_type: generate_until
training_split: train
validation_split: validation
test_split: test
doc_to_text: "Question: {{question}}\nGenerate answer:"
doc_to_target: "{{choices.text[choices.label.index(answerKey)]}}"
generation_kwargs:
  max_gen_toks: 50
  temperature: 0.1
  until:
    - "\n"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
  custom_benchmark: true
"""
            # Write the new task file
            new_task_path = Path(custom_dir) / "arc_custom_generation.yaml"
            new_task_path.write_text(new_task_content)

            # Initialize with include_defaults=True (default behavior)
            task_manager = TaskManager(include_defaults=True, include_path=custom_dir)

            # Both custom and default tasks should be available
            assert "arc_custom_generation" in task_manager.all_tasks, (
                "New custom task should be available"
            )
            assert "arc_easy" in task_manager.all_tasks, (
                "Default arc_easy should still be available"
            )
            assert "arc_challenge" in task_manager.all_tasks, (
                "Default arc_challenge should still be available"
            )

            # Check task metadata using Entry object API
            entry = task_manager.task_index["arc_custom_generation"]
            assert entry.kind == Kind.TASK
            assert custom_dir in str(entry.yaml_path)

            # Verify the counts - should have more tasks than just defaults
            assert len(task_manager.all_tasks) > len(shared_task_manager.all_tasks), (
                "Should have more tasks when including custom path"
            )
