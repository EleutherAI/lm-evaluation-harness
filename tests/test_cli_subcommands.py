import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from lm_eval._cli.harness import HarnessCLI
from lm_eval._cli.ls import List
from lm_eval._cli.run import Run
from lm_eval._cli.utils import (
    MergeDictAction,
    _int_or_none_list_arg_type,
    check_argument_types,
    request_caching_arg_to_dict,
    try_parse_json,
)
from lm_eval._cli.validate import Validate


class TestHarnessCLI:
    """Test the main HarnessCLI class."""

    def test_harness_cli_init(self):
        """Test HarnessCLI initialization."""
        cli = HarnessCLI()
        assert cli._parser is not None
        assert cli._subparsers is not None

    def test_harness_cli_has_subcommands(self):
        """Test that HarnessCLI has all expected subcommands."""
        cli = HarnessCLI()
        subcommands = cli._subparsers.choices
        assert "run" in subcommands
        assert "ls" in subcommands
        assert "validate" in subcommands

    def test_harness_cli_backward_compatibility(self):
        """Test backward compatibility: inserting 'run' when no subcommand is provided."""
        cli = HarnessCLI()
        test_args = ["lm-eval", "--model", "hf", "--tasks", "hellaswag"]
        with patch.object(sys, "argv", test_args):
            args = cli.parse_args()
            assert args.command == "run"
            assert args.model == "hf"
            assert args.tasks == ["hellaswag"]

    def test_harness_cli_help_default(self):
        """Test that help is printed when no arguments are provided."""
        cli = HarnessCLI()
        with patch.object(sys, "argv", ["lm-eval"]):
            args = cli.parse_args()
            # The func is a lambda that calls print_help
            # Let's test it calls the help function correctly
            with patch.object(cli._parser, "print_help") as mock_help:
                args.func(args)
                mock_help.assert_called_once()

    def test_harness_cli_run_help_only(self):
        """Test that 'lm-eval run' shows help."""
        cli = HarnessCLI()
        with patch.object(sys, "argv", ["lm-eval", "run"]), pytest.raises(SystemExit):
            cli.parse_args()


class TestListCommand:
    """Test the List subcommand."""

    def test_list_command_creation(self):
        """Test List command creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        list_cmd = List.create(subparsers)
        assert isinstance(list_cmd, List)

    def test_list_command_arguments(self):
        """Test List command arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        List.create(subparsers)

        # Test valid arguments
        args = parser.parse_args(["ls", "tasks"])
        assert args.what == "tasks"
        assert args.include_path is None

        args = parser.parse_args(["ls", "groups", "--include_path", "/path/to/tasks"])
        assert args.what == "groups"
        assert args.include_path == "/path/to/tasks"

    def test_list_command_choices(self):
        """Test List command only accepts valid choices."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        List.create(subparsers)

        # Valid choices should work
        for choice in ["tasks", "groups", "subtasks", "tags"]:
            args = parser.parse_args(["ls", choice])
            assert args.what == choice

        # Invalid choice should fail
        with pytest.raises(SystemExit):
            parser.parse_args(["ls", "invalid"])

    @patch("lm_eval.tasks.TaskManager")
    def test_list_command_execute_tasks(self, mock_task_manager):
        """Test List command execution for tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        list_cmd = List.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.list_all_tasks.return_value = "task1\ntask2\ntask3"
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["ls", "tasks"])
        with patch("builtins.print") as mock_print:
            list_cmd._execute(args)
            mock_print.assert_called_once_with("task1\ntask2\ntask3")
            mock_tm_instance.list_all_tasks.assert_called_once_with()

    @patch("lm_eval.tasks.TaskManager")
    def test_list_command_execute_groups(self, mock_task_manager):
        """Test List command execution for groups."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        list_cmd = List.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.list_all_tasks.return_value = "group1\ngroup2"
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["ls", "groups"])
        with patch("builtins.print") as mock_print:
            list_cmd._execute(args)
            mock_print.assert_called_once_with("group1\ngroup2")
            mock_tm_instance.list_all_tasks.assert_called_once_with(
                list_subtasks=False, list_tags=False
            )


class TestRunCommand:
    """Test the Run subcommand."""

    def test_run_command_creation(self):
        """Test Run command creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        run_cmd = Run.create(subparsers)
        assert isinstance(run_cmd, Run)

    def test_run_command_basic_arguments(self):
        """Test Run command basic arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        args = parser.parse_args(
            ["run", "--model", "hf", "--tasks", "hellaswag", "arc_easy"]
        )
        assert args.model == "hf"
        assert args.tasks == ["hellaswag", "arc_easy"]

    def test_run_command_tasks_comma_separated(self):
        """Test Run command with comma-separated tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Comma-separated format: --tasks hellaswag,arc_easy
        args = parser.parse_args(["run", "--tasks", "hellaswag,arc_easy"])
        # argparse returns ["hellaswag,arc_easy"], splitting happens in process_tasks
        assert args.tasks == ["hellaswag", "arc_easy"]

    def test_run_command_tasks_mixed_format(self):
        """Test Run command with mixed comma and space-separated tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Mixed format: --tasks hellaswag,arc_easy winogrande
        args = parser.parse_args(["run", "--tasks", "hellaswag,arc_easy", "winogrande"])
        assert args.tasks == ["hellaswag", "arc_easy", "winogrande"]

    def test_run_command_tasks_None(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Mixed format: --tasks hellaswag,arc_easy winogrande
        args = parser.parse_args(["run", "--model", "hf"])
        assert args.tasks is None

    def test_run_command_model_args(self):
        """Test Run command model arguments parsing with MergeDictAction."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test key=value format - MergeDictAction merges into single dict
        args = parser.parse_args(["run", "--model_args", "pretrained=gpt2,device=cuda"])
        assert args.model_args == {"pretrained": "gpt2", "device": "cuda"}

        # Test space-separated key=value pairs - also merged into single dict
        args = parser.parse_args(
            ["run", "--model_args", "pretrained=gpt2", "device=cuda"]
        )
        assert args.model_args == {"pretrained": "gpt2", "device": "cuda"}

    def test_run_command_batch_size(self):
        """Test Run command batch size arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test integer batch size
        args = parser.parse_args(["run", "--batch_size", "32"])
        assert args.batch_size == "32"

        # Test auto batch size
        args = parser.parse_args(["run", "--batch_size", "auto"])
        assert args.batch_size == "auto"

        # Test auto with repetitions
        args = parser.parse_args(["run", "--batch_size", "auto:5"])
        assert args.batch_size == "auto:5"

    def test_run_command_seed_parsing(self):
        """Test Run command seed parsing."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test single seed
        args = parser.parse_args(["run", "--seed", "42"])
        assert args.seed == [42, 42, 42, 42]

        # Test multiple seeds
        args = parser.parse_args(["run", "--seed", "0,1234,5678,9999"])
        assert args.seed == [0, 1234, 5678, 9999]

        # Test with None values
        args = parser.parse_args(["run", "--seed", "0,None,1234,None"])
        assert args.seed == [0, None, 1234, None]

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.config.evaluate_config.EvaluatorConfig")
    @patch("lm_eval.loggers.EvaluationTracker")
    @patch("lm_eval.utils.make_table")
    def test_run_command_execute_basic(
        self, mock_make_table, mock_tracker, mock_config, mock_simple_evaluate
    ):
        """Test Run command basic execution."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        run_cmd = Run.create(subparsers)

        # Mock configuration
        mock_cfg_instance = MagicMock()
        mock_cfg_instance.wandb_args = None
        mock_cfg_instance.output_path = None
        mock_cfg_instance.hf_hub_log_args = {}
        mock_cfg_instance.include_path = None
        mock_cfg_instance.tasks = ["hellaswag"]
        mock_cfg_instance.model = "hf"
        mock_cfg_instance.model_args = {"pretrained": "gpt2"}
        mock_cfg_instance.gen_kwargs = {}
        mock_cfg_instance.limit = None
        mock_cfg_instance.num_fewshot = 0
        mock_cfg_instance.batch_size = 1
        mock_cfg_instance.log_samples = False
        mock_cfg_instance.process_tasks.return_value = MagicMock()
        mock_config.from_cli.return_value = mock_cfg_instance

        # Mock evaluation results
        mock_simple_evaluate.return_value = {
            "results": {"hellaswag": {"acc": 0.75}},
            "config": {"batch_sizes": [1]},
            "configs": {"hellaswag": {}},
            "versions": {"hellaswag": "1.0"},
            "n-shot": {"hellaswag": 0},
        }

        # Mock make_table to avoid complex table rendering
        mock_make_table.return_value = (
            "| Task | Result |\n|------|--------|\n| hellaswag | 0.75 |"
        )

        args = parser.parse_args(["run", "--model", "hf", "--tasks", "hellaswag"])

        with patch("builtins.print"):
            run_cmd._execute(args)

        mock_config.from_cli.assert_called_once()
        mock_simple_evaluate.assert_called_once()
        mock_make_table.assert_called_once()


class TestValidateCommand:
    """Test the Validate subcommand."""

    def test_validate_command_creation(self):
        """Test Validate command creation."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        validate_cmd = Validate.create(subparsers)
        assert isinstance(validate_cmd, Validate)

    def test_validate_command_arguments(self):
        """Test Validate command arguments."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Validate.create(subparsers)

        args = parser.parse_args(["validate", "--tasks", "hellaswag,arc_easy"])
        assert args.tasks == "hellaswag,arc_easy"
        assert args.include_path is None

        args = parser.parse_args(
            ["validate", "--tasks", "custom_task", "--include_path", "/path/to/tasks"]
        )
        assert args.tasks == "custom_task"
        assert args.include_path == "/path/to/tasks"

    def test_validate_command_requires_tasks(self):
        """Test Validate command requires tasks argument."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Validate.create(subparsers)

        with pytest.raises(SystemExit):
            parser.parse_args(["validate"])

    @patch("lm_eval.tasks.TaskManager")
    def test_validate_command_execute_success(self, mock_task_manager):
        """Test Validate command execution with valid tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        validate_cmd = Validate.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.return_value = ["hellaswag", "arc_easy"]
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["validate", "--tasks", "hellaswag,arc_easy"])

        with patch("builtins.print") as mock_print:
            validate_cmd._execute(args)

        mock_print.assert_any_call("Validating tasks: ['hellaswag', 'arc_easy']")
        mock_print.assert_any_call("All tasks found and valid")

    @patch("lm_eval.tasks.TaskManager")
    def test_validate_command_execute_missing_tasks(self, mock_task_manager):
        """Test Validate command execution with missing tasks."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        validate_cmd = Validate.create(subparsers)

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.return_value = ["hellaswag"]
        mock_task_manager.return_value = mock_tm_instance

        args = parser.parse_args(["validate", "--tasks", "hellaswag,nonexistent"])

        with (
            patch("builtins.print") as mock_print,
            pytest.raises(SystemExit) as exc_info,
        ):
            validate_cmd._execute(args)

        assert exc_info.value.code == 1
        mock_print.assert_any_call("Tasks not found: nonexistent")


class TestEvaluatorConfigTaskLoading:
    """Test EvaluatorConfig task loading"""

    @patch("lm_eval.tasks.TaskManager")
    def test_process_tasks_comma_separated_in_list(self, mock_task_manager):
        """Test that process_tasks splits comma-separated tasks from CLI (nargs='*')."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.side_effect = lambda x: x
        mock_task_manager.return_value = mock_tm_instance

        # Simulate CLI input: --tasks hellaswag,arc_easy
        # argparse with nargs="*" gives ["hellaswag,arc_easy"]
        cfg = EvaluatorConfig(tasks=["hellaswag,arc_easy"])
        cfg._configure()
        cfg.process_tasks()

        # Should have split the comma-separated string
        assert mock_tm_instance.match_tasks.call_count == 2
        mock_tm_instance.match_tasks.assert_any_call(["hellaswag"])
        mock_tm_instance.match_tasks.assert_any_call(["arc_easy"])

    @patch("lm_eval.tasks.TaskManager")
    def test_process_tasks_mixed_comma_and_space_separated(self, mock_task_manager):
        """Test process_tasks handles mixed comma and space-separated tasks."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.side_effect = lambda x: x
        mock_task_manager.return_value = mock_tm_instance

        # Simulate CLI input: --tasks hellaswag,arc_easy winogrande
        # argparse with nargs="*" gives ["hellaswag,arc_easy", "winogrande"]
        cfg = EvaluatorConfig(tasks=["hellaswag,arc_easy", "winogrande"])
        cfg._configure()
        cfg.process_tasks()

        # Should have split comma-separated and kept space-separated
        assert mock_tm_instance.match_tasks.call_count == 3
        mock_tm_instance.match_tasks.assert_any_call(["hellaswag"])
        mock_tm_instance.match_tasks.assert_any_call(["arc_easy"])
        mock_tm_instance.match_tasks.assert_any_call(["winogrande"])

    @patch("lm_eval.tasks.TaskManager")
    def test_process_tasks_string_comma_separated(self, mock_task_manager):
        """Test process_tasks splits comma-separated string (from YAML)."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        mock_tm_instance = MagicMock()
        mock_tm_instance.match_tasks.side_effect = lambda x: x
        mock_task_manager.return_value = mock_tm_instance

        # Simulate YAML input: tasks: "hellaswag,arc_easy"
        cfg = EvaluatorConfig(tasks="hellaswag,arc_easy")
        cfg._configure()
        cfg.process_tasks()

        # Should have split the comma-separated string
        assert mock_tm_instance.match_tasks.call_count == 2
        mock_tm_instance.match_tasks.assert_any_call(["hellaswag"])
        mock_tm_instance.match_tasks.assert_any_call(["arc_easy"])

    def test_custom_yaml_file_relative_path(self, tmp_path):
        """Test loading custom task config via a relative path (fixes #3425)."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Create a minimal valid task yaml
        task_yaml = tmp_path / "test_task.yaml"
        task_yaml.write_text("""
task: test_custom_task
dataset_path: hellaswag
output_type: multiple_choice
doc_to_text: "{{question}}"
doc_to_target: "{{answer}}"
""")

        # Test with relative-style path
        cfg = EvaluatorConfig(
            tasks=[str(task_yaml)],
            output_path=str(tmp_path),
        )
        cfg._configure()
        tm = cfg.process_tasks()  # noqa: F841

        # Should load successfully as a dict config, not raise "Tasks not found"
        assert len(cfg.tasks) == 1
        assert isinstance(cfg.tasks[0], dict)
        assert cfg.tasks[0]["task"] == "test_custom_task"

    def test_missing_yaml_file_raises_error(self, tmp_path):
        """Test that non-existent yaml file raises proper error."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        cfg = EvaluatorConfig(
            tasks=[str(tmp_path / "nonexistent.yaml")],
            output_path=str(tmp_path),
        )
        cfg._configure()

        with pytest.raises(ValueError, match="Tasks not found"):
            cfg.process_tasks()


class TestEvaluatorConfigFromCLI:
    """Test EvaluatorConfig.from_cli defaults and argument handling."""

    def test_defaults_applied(self, tmp_path):
        """Test that dataclass defaults are applied when CLI args are missing."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Minimal namespace with only required fields
        ns = Namespace(
            tasks=["hellaswag"],
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        # Check defaults from dataclass
        assert cfg.model == "hf"
        assert cfg.batch_size == 1
        assert cfg.device == "cuda:0"
        assert cfg.num_fewshot is None
        assert cfg.limit is None
        assert cfg.seed == [0, 1234, 1234, 1234]
        assert cfg.trust_remote_code is False
        assert cfg.apply_chat_template is False

    def test_cli_args_override_defaults(self, tmp_path):
        """Test that CLI args override dataclass defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["arc_easy"],
            model="vllm",
            batch_size=16,
            device="cuda:1",
            num_fewshot=5,
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.model == "vllm"
        assert cfg.batch_size == 16
        assert cfg.device == "cuda:1"
        assert cfg.num_fewshot == 5

    def test_model_args_dict_passed_through(self, tmp_path):
        """Test that model_args dict is passed through correctly."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            model_args={"pretrained": "gpt2", "dtype": "float16"},
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.model_args["pretrained"] == "gpt2"
        assert cfg.model_args["dtype"] == "float16"

    def test_gen_kwargs_passed_through(self, tmp_path):
        """Test that gen_kwargs dict is passed through correctly."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            gen_kwargs={"temperature": 0.7, "max_tokens": 100},
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)
        assert cfg.gen_kwargs is not None
        assert cfg.gen_kwargs["temperature"] == 0.7
        assert cfg.gen_kwargs["max_tokens"] == 100

    def test_none_args_use_defaults(self, tmp_path):
        """Test that None values fall back to defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            model=None,  # Should use default "hf"
            batch_size=None,  # Should use default 1
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.model == "hf"
        assert cfg.batch_size == 1

    def test_fewshot_as_multiturn_defaults_with_chat_template(self, tmp_path):
        """Test fewshot_as_multiturn defaults to True when apply_chat_template is set."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            apply_chat_template=True,
            output_path=str(tmp_path),
            log_samples=True,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        assert cfg.fewshot_as_multiturn is True

    def test_empty_tasks_allowed_at_config_level(self):
        """Test that empty tasks passes config validation (fails later in process_tasks)."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        # tasks=None gets filtered out, default [] is used
        ns = Namespace(tasks=None)

        cfg = EvaluatorConfig.from_cli(ns)
        assert cfg.tasks == []  # Empty list, not None

    def test_validation_error_log_samples_without_output(self):
        """Test that log_samples without output_path raises ValueError."""
        from argparse import Namespace

        import pytest

        from lm_eval.config.evaluate_config import EvaluatorConfig

        ns = Namespace(
            tasks=["hellaswag"],
            log_samples=True,
            output_path=None,
        )

        with pytest.raises(ValueError, match="output_path"):
            EvaluatorConfig.from_cli(ns)


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_try_parse_json_with_json_string(self):
        """Test try_parse_json with a valid JSON string."""
        result = try_parse_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_try_parse_json_with_dict(self):
        """Test try_parse_json with dict input."""
        input_dict = {"key": "value"}
        result = try_parse_json(input_dict)
        assert result is input_dict

    def test_try_parse_json_with_none(self):
        """Test try_parse_json with None."""
        result = try_parse_json(None)
        assert result is None

    def test_try_parse_json_with_plain_string(self):
        """Test try_parse_json with a plain string."""
        result = try_parse_json("key=value,key2=value2")
        assert result == "key=value,key2=value2"

    def test_try_parse_json_with_invalid_json(self):
        """Test try_parse_json with invalid JSON."""
        with pytest.raises(ValueError) as exc_info:
            try_parse_json('{key: "value"}')  # Invalid JSON (unquoted key)
        assert "Invalid JSON" in str(exc_info.value)
        assert "double quotes" in str(exc_info.value)

    def test_int_or_none_list_single_value(self):
        """Test _int_or_none_list_arg_type with a single value."""
        result = _int_or_none_list_arg_type(3, 4, "0,1,2,3", "42")
        assert result == [42, 42, 42, 42]

    def test_int_or_none_list_multiple_values(self):
        """Test _int_or_none_list_arg_type with multiple values."""
        result = _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,20,30,40")
        assert result == [10, 20, 30, 40]

    def test_int_or_none_list_with_none(self):
        """Test _int_or_none_list_arg_type with None values."""
        result = _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,None,30,None")
        assert result == [10, None, 30, None]

    def test_int_or_none_list_invalid_value(self):
        """Test _int_or_none_list_arg_type with invalid value."""
        with pytest.raises(ValueError):
            _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,invalid,30,40")

    def test_int_or_none_list_too_few_values(self):
        """Test _int_or_none_list_arg_type with too few values."""
        with pytest.raises(ValueError):
            _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,20")

    def test_int_or_none_list_too_many_values(self):
        """Test _int_or_none_list_arg_type with too many values."""
        with pytest.raises(ValueError):
            _int_or_none_list_arg_type(3, 4, "0,1,2,3", "10,20,30,40,50")

    def test_request_caching_arg_to_dict_none(self):
        """Test request_caching_arg_to_dict with None."""
        result = request_caching_arg_to_dict(None)
        assert result == {}

    def test_request_caching_arg_to_dict_true(self):
        """Test request_caching_arg_to_dict with 'true'."""
        result = request_caching_arg_to_dict("true")
        assert result == {
            "cache_requests": True,
            "rewrite_requests_cache": False,
            "delete_requests_cache": False,
        }

    def test_request_caching_arg_to_dict_refresh(self):
        """Test request_caching_arg_to_dict with 'refresh'."""
        result = request_caching_arg_to_dict("refresh")
        assert result == {
            "cache_requests": True,
            "rewrite_requests_cache": True,
            "delete_requests_cache": False,
        }

    def test_request_caching_arg_to_dict_delete(self):
        """Test request_caching_arg_to_dict with 'delete'."""
        result = request_caching_arg_to_dict("delete")
        assert result == {
            "cache_requests": False,
            "rewrite_requests_cache": False,
            "delete_requests_cache": True,
        }

    def test_request_caching_arg_to_dict_invalid(self):
        """Test request_caching_arg_to_dict rejects invalid values."""
        with pytest.raises(argparse.ArgumentTypeError):
            request_caching_arg_to_dict("bogus")

    def test_cache_requests_argparse_integration(self):
        """Test --cache_requests works end-to-end through argparse.

        Regression test: the `type` function converts the string to a dict
        before `choices` validation, so `choices` must not be used alongside
        `type=request_caching_arg_to_dict`.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cache_requests",
            type=request_caching_arg_to_dict,
            default=None,
        )
        for val in ("true", "refresh", "delete"):
            args = parser.parse_args(["--cache_requests", val])
            assert isinstance(args.cache_requests, dict)

    def test_check_argument_types_raises_on_untyped(self):
        """Test check_argument_types raises error for untyped arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--untyped")  # No type specified

        with pytest.raises(ValueError) as exc_info:
            check_argument_types(parser)
        assert "untyped" in str(exc_info.value)
        assert "doesn't have a type specified" in str(exc_info.value)

    def test_check_argument_types_passes_on_typed(self):
        """Test check_argument_types passes for typed arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--typed", type=str)

        # Should not raise
        check_argument_types(parser)

    def test_check_argument_types_skips_const_actions(self):
        """Test check_argument_types skips const actions."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--flag", action="store_const", const=True)

        # Should not raise
        check_argument_types(parser)


class TestMergeDictAction:
    """Test MergeDictAction for parsing key=value arguments."""

    def test_comma_separated_key_value(self):
        """Test parsing comma-separated key=value pairs."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key1=val1,key2=val2"])
        assert args.args == {"key1": "val1", "key2": "val2"}

    def test_space_separated_key_value(self):
        """Test parsing space-separated key=value pairs."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key1=val1", "key2=val2"])
        assert args.args == {"key1": "val1", "key2": "val2"}

    def test_json_dict_input(self):
        """Test parsing JSON dict input."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", '{"key1": "val1", "key2": 42}'])
        assert args.args == {"key1": "val1", "key2": 42}

    def test_json_nested_dict(self):
        """Test parsing nested JSON dict."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", '{"outer": {"inner": "value"}}'])
        assert args.args == {"outer": {"inner": "value"}}

    def test_empty_values(self):
        """Test that empty values result in None"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="*", action=MergeDictAction)

        args = parser.parse_args(["--args"])
        assert args.args is None

    def test_type_coercion(self):
        """Test that values are coerced to appropriate types."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "num=42,flag=true,pi=3.14"])
        assert args.args["num"] == 42
        assert args.args["flag"] is True
        assert args.args["pi"] == 3.14

    def test_multiple_invocations_merge(self):
        """Test that multiple --args invocations merge values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key1=val1", "--args", "key2=val2"])
        assert args.args == {"key1": "val1", "key2": "val2"}

    def test_key_overwrite(self):
        """Test that later values overwrite earlier ones."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--args", nargs="+", action=MergeDictAction)

        args = parser.parse_args(["--args", "key=first", "--args", "key=second"])
        assert args.args["key"] == "second"


class TestEvaluatorConfigPrecedence:
    """Test EvaluatorConfig merging precedence: CLI args > YAML config > built-in defaults."""

    def test_cli_overrides_yaml_overrides_defaults(self, tmp_path):
        """Test full precedence chain: CLI args > YAML config > built-in defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Create a YAML config file
        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("""
model: vllm
batch_size: 8
device: cuda:1
num_fewshot: 3
tasks:
  - hellaswag
output_path: /yaml/path
log_samples: true
""")

        # CLI args: override some YAML values, leave others to YAML/defaults
        ns = Namespace(
            config=str(yaml_config),
            model="openai",  # CLI overrides YAML's "vllm"
            batch_size=32,  # CLI overrides YAML's 8
            # device not specified in CLI -> should use YAML's cuda:1
            # num_fewshot not specified in CLI -> should use YAML's 3
            tasks=None,  # falsy, should use YAML's hellaswag
            output_path=None,  # falsy, should use YAML's /yaml/path
            log_samples=None,  # falsy, should use YAML's true
        )

        cfg = EvaluatorConfig.from_cli(ns)

        # CLI values win
        assert cfg.model == "openai", "CLI should override YAML"
        assert cfg.batch_size == 32, "CLI should override YAML"

        # YAML values win over defaults
        assert cfg.device == "cuda:1", "YAML should override default (cuda:0)"
        assert cfg.num_fewshot == 3, "YAML should override default (None)"
        assert cfg.tasks == ["hellaswag"], "YAML should be used when CLI is falsy"
        assert cfg.output_path == "/yaml/path", "YAML should be used when CLI is falsy"

        # Defaults used when neither CLI nor YAML specify
        assert cfg.trust_remote_code is False, "Should use default"
        assert cfg.seed == [0, 1234, 1234, 1234], "Should use default"

    def test_yaml_overrides_defaults(self, tmp_path):
        """Test that YAML config values override built-in defaults."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("""
model: vllm
batch_size: 16
device: cpu
seed: [42, 42, 42, 42]
tasks:
  - arc_easy
output_path: /tmp/out
log_samples: true
""")

        ns = Namespace(config=str(yaml_config))

        cfg = EvaluatorConfig.from_cli(ns)

        # All should come from YAML, not defaults
        assert cfg.model == "vllm", "YAML should override default 'hf'"
        assert cfg.batch_size == 16, "YAML should override default 1"
        assert cfg.device == "cpu", "YAML should override default 'cuda:0'"
        assert cfg.seed == [42, 42, 42, 42], "YAML should override default seed"

    def test_cli_overrides_yaml_with_explicit_zero(self, tmp_path):
        """Test that explicit CLI value 0 overrides YAML."""
        from argparse import Namespace

        from lm_eval.config.evaluate_config import EvaluatorConfig

        yaml_config = tmp_path / "config.yaml"
        yaml_config.write_text("""
model: vllm
batch_size: 16
num_fewshot: 5
tasks:
  - hellaswag
output_path: /yaml/path
log_samples: true
""")

        ns = Namespace(
            config=str(yaml_config),
            num_fewshot=0,  # Explicit 0 should override YAML's 5
            batch_size=1,  # Explicit 1 should override
            tasks=None,
            output_path=None,
            log_samples=None,
        )

        cfg = EvaluatorConfig.from_cli(ns)

        # 0 is a valid explicit value and should override YAML
        assert cfg.num_fewshot == 0, "CLI 0 should override YAML's 5"
        assert cfg.batch_size == 1, "Truthy CLI value overrides YAML"
