#!/usr/bin/env python3
"""
Walkthrough tests using real dataset configurations.

These tests use YAML configs with existing datasets (hellaswag) to enable
complete code walkthrough of the task loading system, including:
- Basic task loading
- Task list functionality
- Group functionality
- Include inheritance
- Issue #2158 fix (include processing preserving task names)
"""

import os

import pytest

from lm_eval.tasks import TaskManager, get_task_dict


class TestWalkthroughConfigs:
    """Test walkthrough configurations for easier code demonstration"""

    @pytest.fixture(autouse=True)
    def setup_task_manager(self):
        """Set up TaskManager with test configs directory"""
        test_configs_dir = os.path.join(os.path.dirname(__file__), "test_configs")
        self.tm = TaskManager(include_path=test_configs_dir, include_defaults=False)

    def test_simple_task_loading(self):
        """Test basic task loading - walkthrough starting point"""
        # Simple task should be indexed
        assert "simple_task" in self.tm.all_tasks
        assert self.tm._name_is_task("simple_task")

        # Load the task
        task_dict = get_task_dict(["simple_task"], task_manager=self.tm)
        assert "simple_task" in task_dict

        # Verify task configuration
        task_obj = task_dict["simple_task"]
        assert hasattr(task_obj, "config")
        assert task_obj.config.task == "simple_task"

    def test_task_list_functionality(self):
        """Test task_list feature - multiple tasks sharing config"""

        # All task_list tasks should be indexed as individual tasks
        expected_tasks = ["task_list_fs0", "task_list_fs1", "task_list_fs3"]

        for task_name in expected_tasks:
            assert task_name in self.tm.all_tasks, f"Task {task_name} not indexed"
            assert self.tm._name_is_task(task_name), (
                f"Task {task_name} not recognized as task"
            )

        # Load all tasks from the task_list
        task_dict = get_task_dict(expected_tasks, task_manager=self.tm)

        # Each should be a separate task object
        assert len(task_dict) == 3
        for task_name in expected_tasks:
            assert task_name in task_dict
            task_obj = task_dict[task_name]
            assert task_obj.config.task == task_name

        # Verify different num_fewshot values were applied
        assert task_dict["task_list_fs0"].config.num_fewshot == 0
        assert task_dict["task_list_fs1"].config.num_fewshot == 1
        assert task_dict["task_list_fs3"].config.num_fewshot == 3

    def test_group_functionality(self):
        """Test group loading with task-specific overrides"""

        # Group should be indexed
        assert "test_group" in self.tm.all_groups
        assert self.tm._name_is_group("test_group")

        # Load the group
        task_dict = get_task_dict(["test_group"], task_manager=self.tm)

        # Should contain the group object and its subtasks
        assert len(task_dict) == 1
        group_obj = list(task_dict.keys())[0]
        subtasks = task_dict[group_obj]

        # Check expected subtasks
        expected_subtasks = ["group_task_fs0", "group_task_fs2"]
        for subtask_name in expected_subtasks:
            assert subtask_name in subtasks

        # Verify different configurations were applied
        fs0_task = subtasks["group_task_fs0"]
        fs2_task = subtasks["group_task_fs2"]
        assert fs0_task.config.num_fewshot == 0
        assert fs2_task.config.num_fewshot == 2

    def test_include_inheritance(self):
        """Test include functionality and inheritance"""

        # Test direct include tasks (these were created as separate files)
        include_tasks = ["include_task_fs0", "include_task_fs1", "include_task_fs5"]

        for task_name in include_tasks:
            assert task_name in self.tm.all_tasks

        # Load tasks that use include
        task_dict = get_task_dict(
            include_tasks[:1], task_manager=self.tm
        )  # Just test first one

        # Should inherit from base config
        task_obj = task_dict["include_task_fs0"]
        # Should inherit dataset_path from include
        assert task_obj.config.dataset_path == "json"
        # Should inherit output_type from include
        assert task_obj.config.output_type == "multiple_choice"
        # Should preserve specific task name (not base_task_name)
        assert task_obj.config.task == "include_task_fs0"
        # Should have overridden num_fewshot
        assert task_obj.config.num_fewshot == 0

    def test_issue_2158_fix_demo(self):
        """
        Test issue #2158 fix - multiple tasks with same include in group.

        This demonstrates the specific scenario that was failing before the fix.
        """

        # Group with multiple tasks using same include should work
        assert "include_group" in self.tm.all_groups

        # This should NOT raise a duplicate detection error
        # Before the fix, this would fail with:
        # "Please call groups which overlap their constituent tasks in separate evaluation runs"
        task_dict = get_task_dict(["include_group"], task_manager=self.tm)

        # Should successfully load the group
        assert len(task_dict) == 1
        group_obj = list(task_dict.keys())[0]
        subtasks = task_dict[group_obj]

        # Check all expected tasks are present with correct names
        expected_tasks = ["include_task_fs0", "include_task_fs1", "include_task_fs5"]

        for task_name in expected_tasks:
            assert task_name in subtasks, f"Task {task_name} missing from group"
            task_obj = subtasks[task_name]

            # CRITICAL: Task name should be preserved, not overwritten by include
            assert task_obj.config.task == task_name

            # Should inherit base config from include
            assert task_obj.config.dataset_path == "json"
            assert task_obj.config.output_type == "multiple_choice"

        # Verify different num_fewshot values
        assert subtasks["include_task_fs0"].config.num_fewshot == 0
        assert subtasks["include_task_fs1"].config.num_fewshot == 1
        assert subtasks["include_task_fs5"].config.num_fewshot == 5

    def test_config_types_detection(self):
        """Test that different config types are correctly detected"""

        # Load various config types to test detection methods
        configs = [
            # Simple task config
            {"task": "walkthrough_simple_task"},
            # Group config
            {"group": "test_group", "task": ["task1", "task2"]},
            # Task list config (would need to be loaded from file)
        ]

        # Test config detection methods
        assert self.tm._config_is_task(configs[0])
        assert not self.tm._config_is_group()
        assert not self.tm._config_is_task_list(configs[0])

        assert not self.tm._config_is_task(configs[1])
        assert self.tm._config_is_group()
        assert not self.tm._config_is_task_list(configs[1])

        # Test task_list detection with actual config
        task_list_config = {"task_list": [{"task": "task1"}, {"task": "task2"}]}
        assert self.tm._config_is_task_list(task_list_config)
        assert not self.tm._config_is_task(task_list_config)
        assert not self.tm._config_is_group()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
