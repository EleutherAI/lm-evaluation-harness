import logging
import tempfile
from pathlib import Path

import pytest

from lm_eval.tasks import TaskManager
from lm_eval.tasks._index import Entry, Kind, TaskIndex
from lm_eval.tasks._yaml_loader import load_yaml


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
from lm_eval.api.task import ConfigurableTask

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
        """Verify debug message logged for duplicate group names"""
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
        with caplog.at_level(logging.DEBUG):
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
    """Create a TaskManager with default tasks"""
    return TaskManager()


@pytest.fixture(scope="module")
def test_configs_task_manager():
    """TaskManager with only test_configs tasks"""
    test_configs_path = Path(__file__).parent / "test_configs"
    return TaskManager(include_path=str(test_configs_path), include_defaults=False)


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
            assert (
                shared_task_manager._entry(g) is not None
                and shared_task_manager._entry(g).kind == Kind.GROUP
            )

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
            assert (
                shared_task_manager._entry(t) is not None
                and shared_task_manager._entry(t).kind == Kind.TAG
            )

    def test_load_task_by_name(self, test_configs_task_manager):
        """Load a single task by name"""
        result = test_configs_task_manager.load_task_or_group(["simple_task"])
        assert "simple_task" in result

    def test_load_group_by_name(self, test_configs_task_manager):
        """Load a group and get nested structure with namespaced task names"""
        result = test_configs_task_manager.load_task_or_group(["test_group"])
        # Result is {ConfigurableGroup: {task_name: task_obj}}
        # Get the children dict from the group
        children = list(result.values())[0]
        # test_group contains inline tasks, namespaced as group_name::task_name
        assert "test_group::group_task_fs0" in children
        assert "test_group::group_task_fs2" in children

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

    def test_match_tasks_glob(self, shared_task_manager):
        """match_tasks handles glob patterns"""
        matches = shared_task_manager.match_tasks(["arc_*"])
        assert "arc_easy" in matches
        assert "arc_challenge" in matches

    def test_name_is_registered(self, shared_task_manager):
        """_name_is_registered checks if name exists"""
        assert "arc_easy" in shared_task_manager._index
        assert "nonexistent_task_xyz" not in shared_task_manager._index

    def test_name_is_task_tag(self, shared_task_manager):
        """_name_is_task returns True for tasks"""
        assert "arc_easy" in shared_task_manager._index
        assert shared_task_manager._index["arc_easy"].kind == Kind.TASK
        entry = shared_task_manager._index.get("ai2_arc")
        assert entry is not None
        assert entry.kind == Kind.TAG  # ai2_arc is a tag, not a task

    def test_include_path_precedence(self, shared_task_manager):
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
            # Use shared_task_manager instead of creating a new one (saves ~9s)
            default_task_dict = shared_task_manager.load_task_or_group(["arc_easy"])
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

    def test_tag_expansion_in_group(self, test_configs_task_manager):
        """Test that TAGs inside groups are expanded and each task is namespaced individually.

        This tests the MMLU-like structure: GROUP -> TAG -> multiple tasks
        Without proper TAG handling, all tasks in the tag get the same namespaced name
        and collide, leaving only one task.
        """
        # Load the subgroup that contains a TAG reference
        result = test_configs_task_manager.load_task_or_group(["tag_subgroup"])

        # Get the children dict from the group
        group_key = list(result.keys())[0]
        children = result[group_key]

        # All 3 tasks from the tag should be expanded
        assert "tag_task_1" in children, "tag_task_1 should be in tag_subgroup"
        assert "tag_task_2" in children, "tag_task_2 should be in tag_subgroup"
        assert "tag_task_3" in children, "tag_task_3 should be in tag_subgroup"

        # Verify we have exactly 3 tasks (not 1 due to collision)
        assert len(children) == 3, (
            f"Should have 3 tasks from TAG expansion, got {len(children)}"
        )

    def test_nested_group_with_tag(self, test_configs_task_manager):
        """Test nested groups with TAG: parent_group -> subgroup -> TAG -> tasks.

        This simulates the full MMLU structure where:
        - mmlu (GROUP) contains mmlu_humanities (GROUP)
        - mmlu_humanities contains mmlu_humanities_tasks (TAG)
        - The TAG expands to individual tasks
        """
        # Load the parent group
        result = test_configs_task_manager.load_task_or_group(["tag_parent_group"])

        # Navigate the nested structure
        parent_key = list(result.keys())[0]
        parent_children = result[parent_key]

        # Should contain the subgroup
        assert len(parent_children) == 1, "Parent should have 1 child (the subgroup)"

        # Get the subgroup
        subgroup_key = list(parent_children.keys())[0]
        subgroup_children = parent_children[subgroup_key]

        # The subgroup should have all 3 tasks expanded from the TAG
        # Tasks are namespaced under their immediate parent group (tag_subgroup)
        assert "tag_task_1" in subgroup_children
        assert "tag_task_2" in subgroup_children
        assert "tag_task_3" in subgroup_children
        assert len(subgroup_children) == 3, (
            f"Subgroup should have 3 tasks, got {len(subgroup_children)}"
        )

    def test_inline_subgroup_syntax(self, test_configs_task_manager):
        """Test inline subgroup syntax: task: [{group: name, task: [...]}].

        This is the format used by mmlu_flan_cot_fewshot and similar configs.
        """
        tm = test_configs_task_manager

        # Load the group with inline subgroups
        loaded = tm.load(["inline_subgroup_parent"])

        # Should have one top-level group
        groups = loaded.get("groups", {})
        assert "inline_subgroup_parent" in groups
        parent_group = groups["inline_subgroup_parent"]
        assert parent_group.name == "inline_subgroup_parent"

        # Should have two inline subgroups (use recursive=False for direct children only)
        subgroups = parent_group.get_all_groups(recursive=False)
        assert len(subgroups) == 2

        subgroup_names = {g.name for g in subgroups}
        assert "inline_subgroup_parent::subgroup_a" in subgroup_names
        assert "inline_subgroup_parent::subgroup_b" in subgroup_names

        # Each subgroup should have its tasks
        for subgroup in subgroups:
            assert len(subgroup.get_all_tasks(recursive=False)) == 1
            # Verify aggregation was parsed
            assert subgroup.aggregate_metric_list is not None


# =============================================================================
# Same integration tests using load() instead of load_task_or_group()
# =============================================================================


class TestTaskManagerLoad:
    """Mirror of TestTaskManagerIntegration using the new load() API.

    Verifies that load() returns equivalent data to the deprecated
    load_task_or_group() for every scenario.
    """

    def test_load_task_by_name(self, test_configs_task_manager):
        """Load a single task by name"""
        result = test_configs_task_manager.load(["simple_task"])
        assert "simple_task" in result["tasks"]

    def test_load_group_by_name(self, test_configs_task_manager):
        """Load a group and get tasks + groups dicts"""
        result = test_configs_task_manager.load(["test_group"])
        assert "test_group" in result["groups"]
        tasks = result["tasks"]
        assert "test_group::group_task_fs0" in tasks
        assert "test_group::group_task_fs2" in tasks

    def test_load_group_map(self, test_configs_task_manager):
        """group_map lists direct children of each group"""
        result = test_configs_task_manager.load(["test_group"])
        gm = result["group_map"]
        assert "test_group" in gm
        assert "test_group::group_task_fs0" in gm["test_group"]
        assert "test_group::group_task_fs2" in gm["test_group"]

    def test_load_tag_by_name(self, shared_task_manager):
        """Load all tasks in a tag"""
        result = shared_task_manager.load(["ai2_arc"])
        assert "arc_easy" in result["tasks"]
        assert "arc_challenge" in result["tasks"]
        # Tags don't create groups
        assert len(result["groups"]) == 0

    def test_include_inheritance_override(self):
        """Child config overrides parent values from include"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)

        result = tm.load(["include_task_fs5"])
        task_obj = result["tasks"]["include_task_fs5"]

        assert task_obj.config.num_fewshot == 5
        assert task_obj.config.dataset_path == "json"

    def test_include_custom_metrics(self):
        """include_task_fs5 has custom metrics (acc + acc_norm)"""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(include_path=str(test_configs_path), include_defaults=False)

        result = tm.load(["include_task_fs5"])
        task_obj = result["tasks"]["include_task_fs5"]

        assert task_obj.config.metric_list is not None, "metric_list should not be None"
        metric_names = [m["metric"] for m in task_obj.config.metric_list]
        assert "acc" in metric_names
        assert "acc_norm" in metric_names

    def test_tag_expansion_in_group(self, test_configs_task_manager):
        """TAGs inside groups expand each task individually"""
        result = test_configs_task_manager.load(["tag_subgroup"])
        tasks = result["tasks"]

        assert "tag_task_1" in tasks
        assert "tag_task_2" in tasks
        assert "tag_task_3" in tasks
        assert len(tasks) == 3

    def test_nested_group_with_tag(self, test_configs_task_manager):
        """Nested groups with TAG: parent -> subgroup -> TAG -> tasks"""
        result = test_configs_task_manager.load(["tag_parent_group"])

        groups = result["groups"]
        assert "tag_parent_group" in groups
        assert "tag_subgroup" in groups

        tasks = result["tasks"]
        assert "tag_task_1" in tasks
        assert "tag_task_2" in tasks
        assert "tag_task_3" in tasks

    def test_include_path_precedence(self, shared_task_manager):
        """User-specified include paths override default tasks"""
        with tempfile.TemporaryDirectory() as custom_dir:
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
            (Path(custom_dir) / "arc_easy.yaml").write_text(custom_task_content)

            tm = TaskManager(include_defaults=True, include_path=custom_dir)
            result = tm.load(["arc_easy"])
            arc_easy = result["tasks"]["arc_easy"]

            assert any(m["metric"] == "f1" for m in arc_easy.config["metric_list"])
            assert "Custom Question:" in arc_easy.config["doc_to_text"]

            # Default should not have custom metric
            default_result = shared_task_manager.load(["arc_easy"])
            default_arc = default_result["tasks"]["arc_easy"]
            assert not any(
                m["metric"] == "f1" for m in default_arc.config.get("metric_list", [])
            )

    def test_load_returns_same_tasks_as_legacy(self, test_configs_task_manager):
        """load() and load_task_or_group() produce the same leaf tasks"""
        new = test_configs_task_manager.load(["test_group"])
        with pytest.warns(DeprecationWarning):
            old = test_configs_task_manager.load_task_or_group(["test_group"])

        # Flatten legacy nested dict to get leaf task names
        def _leaf_names(d):
            names = set()
            for k, v in d.items():
                if isinstance(v, dict):
                    names |= _leaf_names(v)
                else:
                    names.add(k)
            return names

        assert set(new["tasks"].keys()) == _leaf_names(old)


# =============================================================================
# Group Building & Factory Tests
# =============================================================================


class TestGroupBuilding:
    """Tests for group building logic in TaskFactory.

    Covers: config propagation, override precedence, inline vs registry groups,
    existing group references with overrides, aggregation parsing, and edge cases.
    """

    @pytest.fixture()
    def tm(self):
        test_configs_path = Path(__file__).parent / "test_configs"
        return TaskManager(include_path=str(test_configs_path), include_defaults=False)

    # ---- existing group reference with overrides (the key bug fix) ----

    def test_existing_group_ref_has_children(self, tm):
        """
        When a parent group references an existing group via
        {group: include_group, ...}, the referenced group must still
        have its own children populated from the registry.
        """
        loaded = tm.load(["group_ref_parent"])
        parent = loaded["groups"]["group_ref_parent"]

        child_groups = parent.get_all_groups(recursive=False)
        assert len(child_groups) == 1

        include_grp = child_groups[0]
        assert include_grp.name == "include_group"

        tasks = include_grp.get_all_tasks(recursive=False)
        assert len(tasks) == 3, (
            f"include_group should have 3 children, got {len(tasks)}: "
            f"{[t.task_name for t in tasks]}"
        )

    def test_existing_group_ref_overrides_propagate(self, tm):
        """
        Overrides specified on a group reference should propagate
        down to the leaf tasks of the referenced group.
        """
        loaded = tm.load(["group_ref_parent"])
        parent = loaded["groups"]["group_ref_parent"]
        include_grp = parent.get_all_groups(recursive=False)[0]

        for task in include_grp.get_all_tasks():
            assert task.config.num_fewshot == 99, (
                f"{task.task_name}: expected num_fewshot=99, "
                f"got {task.config.num_fewshot}"
            )

    # ---- group-level config propagation ----

    def test_group_level_config_propagates_to_children(self, tm):
        """
        Config keys set at the group level (outside GROUP_ONLY_KEYS)
        should propagate to all children as defaults.
        """
        loaded = tm.load(["propagation_group"])
        tasks = loaded["tasks"]

        assert len(tasks) == 2
        for name, task in tasks.items():
            assert task.config.num_fewshot == 42, (
                f"{name}: expected num_fewshot=42 from group, "
                f"got {task.config.num_fewshot}"
            )

    def test_caller_overrides_beat_group_defaults(self, tm):
        """
        Caller-supplied overrides should take precedence over
        group-level config values.
        """
        loaded = tm.load([{"group": "propagation_group", "num_fewshot": 0}])
        tasks = loaded["tasks"]

        for name, task in tasks.items():
            assert task.config.num_fewshot == 0, (
                f"{name}: expected num_fewshot=0 from caller override, "
                f"got {task.config.num_fewshot}"
            )

    # ---- mixed member types ----

    def test_mixed_members_string_ref(self, tm):
        """
        A bare string in the task list should resolve to the task in
        the registry.
        """
        loaded = tm.load(["mixed_members_group"])
        tasks = loaded["tasks"]
        assert "simple_task" in tasks

    def test_mixed_members_dict_with_overrides(self, tm):
        """A dict with 'task' key should apply inline overrides."""
        loaded = tm.load(["mixed_members_group"])
        task_b = loaded["tasks"]["simple_task_b"]
        assert task_b.config.num_fewshot == 7

    def test_mixed_members_inline_subgroup(self, tm):
        """
        A dict with 'group' key (not in registry) should create an
        inline subgroup with namespacing.
        """
        loaded = tm.load(["mixed_members_group"])
        groups = loaded["groups"]

        # The inline subgroup should be namespaced
        assert "mixed_members_group::mixed_inline_sub" in groups
        inline = groups["mixed_members_group::mixed_inline_sub"]
        assert len(inline.get_all_tasks(recursive=False)) == 1
        assert inline.aggregate_metric_list is not None
        assert inline.aggregate_metric_list[0].metric == "acc"

    # ---- empty group ----

    def test_empty_group_has_no_children(self, tm):
        """
        A group with no task list should build successfully with
        zero children.
        """
        loaded = tm.load(["empty_group"])
        group = loaded["groups"]["empty_group"]
        assert len(group) == 0
        assert group.get_all_tasks() == []
        # Aggregation should still be parsed
        assert group.has_aggregation

    # ---- aggregation parsing (unit-level, via GroupConfig __post_init__) ----

    def test_parse_aggregation_with_list(self):
        """aggregate_metric_list as a list should parse to list[AggMetricConfig]."""
        from lm_eval.config.group import AggMetricConfig, GroupConfig

        cfg = GroupConfig(
            group="test",
            aggregate_metric_list=[
                {"metric": "acc", "weight_by_size": True},
                {"metric": "f1", "weight_by_size": False},
            ],
        )
        result = cfg.aggregate_metric_list
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], AggMetricConfig)
        assert result[0].metric == "acc"
        assert result[0].weight_by_size is True
        assert isinstance(result[1], AggMetricConfig)
        assert result[1].metric == "f1"
        assert result[1].weight_by_size is False

    def test_parse_aggregation_single_dict_normalized(self):
        """
        A single dict (not wrapped in a list) should be normalized
        to a one-element list.
        """
        from lm_eval.config.group import AggMetricConfig, GroupConfig

        cfg = GroupConfig(
            group="test",
            aggregate_metric_list={"metric": "acc", "weight_by_size": True},  # type:ignore[invalid-argument-type]
        )
        result = cfg.aggregate_metric_list
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], AggMetricConfig)
        assert result[0].metric == "acc"

    def test_parse_aggregation_missing_returns_none(self):
        """No aggregate_metric_list key should return None."""
        from lm_eval.config.group import GroupConfig

        assert GroupConfig(group="test").aggregate_metric_list is None
        assert (
            GroupConfig(group="test", aggregate_metric_list=None).aggregate_metric_list
            is None
        )

    # ---- group alias / metadata ----

    def test_group_alias_preserved(self, tm):
        """group_alias from config should appear on the Group object."""
        # inline_subgroup_parent has no alias, but its children (subgroup_a, subgroup_b) don't either
        # Use include_group which also has no alias — just verify the field exists
        loaded = tm.load(["include_group"])
        group = loaded["groups"]["include_group"]
        # include_group.yaml doesn't set group_alias, so it should be None
        assert group.alias is None

    def test_group_metadata_includes_factory_meta(self):
        """Factory-level metadata should be merged into every group's metadata."""
        test_configs_path = Path(__file__).parent / "test_configs"
        tm = TaskManager(
            include_path=str(test_configs_path),
            include_defaults=False,
            metadata={"run_id": "test-123"},
        )
        loaded = tm.load(["include_group"])
        group = loaded["groups"]["include_group"]
        assert group.metadata is not None
        assert group.metadata.get("run_id") == "test-123"

    # ---- nested groups ----

    def test_deeply_nested_get_all_tasks_recursive(self, tm):
        """
        get_all_tasks(recursive=True) on a parent group should
        collect tasks from all levels of nesting.
        """
        loaded = tm.load(["group_ref_parent"])
        parent = loaded["groups"]["group_ref_parent"]

        # parent -> include_group -> 3 tasks
        all_tasks = parent.get_all_tasks(recursive=True)
        assert len(all_tasks) == 3

    def test_deeply_nested_get_all_tasks_non_recursive(self, tm):
        """
        get_all_tasks(recursive=False) on a parent group should
        NOT return tasks from nested subgroups.
        """
        loaded = tm.load(["group_ref_parent"])
        parent = loaded["groups"]["group_ref_parent"]

        direct_tasks = parent.get_all_tasks(recursive=False)
        assert len(direct_tasks) == 0  # parent only has a subgroup, no direct tasks
