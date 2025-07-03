import argparse
import sys
from typing import Dict, Type

from lm_eval._cli.base import SubCommand
from lm_eval._cli.cache import CacheCommand
from lm_eval._cli.evaluate import EvaluateCommand
from lm_eval._cli.list import ListCommand
from lm_eval._cli.validate import ValidateCommand


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        # Skip help, subcommands, and const actions
        if action.dest in ["help", "command"] or action.const is not None:
            continue
        if action.type is None:
            raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
        else:
            continue


class CLIParser:
    """Main CLI parser class that manages all subcommands."""

    def __init__(self):
        self.parser = None
        self.subparsers = None
        self.legacy_parser = None
        self.command_instances: Dict[str, SubCommand] = {}

    def setup_parser(self) -> argparse.ArgumentParser:
        """Set up the main parser with subcommands."""
        if self.parser is not None:
            return self.parser

        self.parser = argparse.ArgumentParser(
            prog="lm-eval",
            description="Language Model Evaluation Harness",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Create subparsers
        self.subparsers = self.parser.add_subparsers(
            dest="command", help="Available commands", metavar="COMMAND"
        )

        # Create and register all command instances
        self.command_instances = {
            "evaluate": EvaluateCommand.create(self.subparsers),
            "list": ListCommand.create(self.subparsers),
            "validate": ValidateCommand.create(self.subparsers),
            "cache": CacheCommand.create(self.subparsers),
        }

        return self.parser

    def setup_legacy_parser(self) -> argparse.ArgumentParser:
        """Set up legacy parser for backward compatibility."""
        if self.legacy_parser is not None:
            return self.legacy_parser

        self.legacy_parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter
        )

        # For legacy mode, we just need to add the evaluate command's arguments
        # without the subcommand structure. We'll create a temporary instance.
        from lm_eval._cli.evaluate import EvaluateCommand as EvalCmd

        # Create a minimal instance just to get the arguments
        temp_cmd = object.__new__(EvalCmd)
        temp_cmd._add_args(self.legacy_parser)

        return self.legacy_parser

    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse arguments using the main parser."""
        parser = self.setup_parser()
        check_argument_types(parser)
        return parser.parse_args(args)

    def parse_legacy_args(self, args=None) -> argparse.Namespace:
        """Parse arguments using the legacy parser."""
        parser = self.setup_legacy_parser()
        check_argument_types(parser)
        return parser.parse_args(args)

    def should_use_subcommand_mode(self, argv=None) -> bool:
        """Determine if we should use subcommand mode based on arguments."""
        if argv is None:
            argv = sys.argv[1:]

        # If no arguments, show main help
        if len(argv) == 0:
            return True

        # Check if first argument is a known subcommand
        # First ensure parser is set up to populate command_instances
        if not self.command_instances:
            self.setup_parser()

        if len(argv) > 0 and argv[0] in self.command_instances:
            return True

        return False

    def execute(self, argv=None) -> None:
        """Main execution method that handles both subcommand and legacy modes."""
        if self.should_use_subcommand_mode(argv):
            # Use subcommand mode
            if argv is None and len(sys.argv) == 1:
                # No arguments provided, show help
                self.setup_parser().print_help()
                sys.exit(1)

            args = self.parse_args(argv)
            args.func(args)
        else:
            # Use legacy mode for backward compatibility
            args = self.parse_legacy_args(argv)
            self._handle_legacy_mode(args)

    def _handle_legacy_mode(self, args: argparse.Namespace) -> None:
        """Handle legacy CLI mode for backward compatibility."""

        # Handle legacy task listing
        if hasattr(args, "tasks") and args.tasks in [
            "list",
            "list_groups",
            "list_subtasks",
            "list_tags",
        ]:
            from lm_eval.tasks import TaskManager

            task_manager = TaskManager(include_path=getattr(args, "include_path", None))

            if args.tasks == "list":
                print(task_manager.list_all_tasks())
            elif args.tasks == "list_groups":
                print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
            elif args.tasks == "list_subtasks":
                print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
            elif args.tasks == "list_tags":
                print(
                    task_manager.list_all_tasks(list_groups=False, list_subtasks=False)
                )
            sys.exit(0)

        # Handle legacy evaluation
        # Use existing instance if available, otherwise create temporary one
        if "evaluate" in self.command_instances:
            evaluate_cmd = self.command_instances["evaluate"]
        else:
            # For legacy mode, we don't need the subparser registration
            # Just execute with the existing args
            from lm_eval._cli.evaluate import EvaluateCommand as EvalCmd

            # Create a minimal instance just for execution
            evaluate_cmd = object.__new__(EvalCmd)
        evaluate_cmd.execute(args)

    def add_command(self, name: str, command_class: Type[SubCommand]) -> None:
        """Add a new command to the parser (for extensibility)."""
        # If parser is already set up, create and register the command instance
        if self.subparsers is not None:
            self.command_instances[name] = command_class.create(self.subparsers)
        else:
            # Store class for later instantiation
            if not hasattr(self, "_pending_commands"):
                self._pending_commands = {}
            self._pending_commands[name] = command_class
