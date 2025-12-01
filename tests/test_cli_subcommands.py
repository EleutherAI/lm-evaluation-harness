import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from lm_eval._cli.harness import HarnessCLI
from lm_eval._cli.ls import List
from lm_eval._cli.run import Run
from lm_eval._cli.utils import (
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
            assert args.tasks == "hellaswag"

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
            ["run", "--model", "hf", "--tasks", "hellaswag,arc_easy"]
        )
        assert args.model == "hf"
        assert args.tasks == "hellaswag,arc_easy"

    def test_run_command_model_args(self):
        """Test Run command model arguments parsing."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        Run.create(subparsers)

        # Test key=value format
        args = parser.parse_args(["run", "--model_args", "pretrained=gpt2,device=cuda"])
        assert args.model_args == "pretrained=gpt2,device=cuda"

        # Test JSON format
        args = parser.parse_args(
            ["run", "--model_args", '{"pretrained": "gpt2", "device": "cuda"}']
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

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc_info:
                validate_cmd._execute(args)

        assert exc_info.value.code == 1
        mock_print.assert_any_call("Tasks not found: nonexistent")


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_try_parse_json_with_json_string(self):
        """Test try_parse_json with valid JSON string."""
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
        """Test try_parse_json with plain string."""
        result = try_parse_json("key=value,key2=value2")
        assert result == "key=value,key2=value2"

    def test_try_parse_json_with_invalid_json(self):
        """Test try_parse_json with invalid JSON."""
        with pytest.raises(ValueError) as exc_info:
            try_parse_json('{key: "value"}')  # Invalid JSON (unquoted key)
        assert "Invalid JSON" in str(exc_info.value)
        assert "double quotes" in str(exc_info.value)

    def test_int_or_none_list_single_value(self):
        """Test _int_or_none_list_arg_type with single value."""
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
