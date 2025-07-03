import argparse
import sys

from lm_eval._cli.cache import Cache
from lm_eval._cli.run import Run
from lm_eval._cli.list import ListCommand
from lm_eval._cli.validate import ValidateCommand


class CLIParser:
    """Main CLI parser class that manages all subcommands."""

    def __init__(self):
        self._parser = argparse.ArgumentParser(
            prog="lm-eval",
            description="Language Model Evaluation Harness",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._parser.set_defaults(func=lambda args: self._parser.print_help())
        self._subparsers = self._parser.add_subparsers(
            dest="command", help="Available commands", metavar="COMMAND"
        )
        Run.create(self._subparsers)
        ListCommand.create(self._subparsers)
        ValidateCommand.create(self._subparsers)
        Cache.create(self._subparsers)

    def parse_args(self) -> argparse.Namespace:
        """Parse arguments using the main parser."""
        if len(sys.argv) > 2 and sys.argv[1] not in self._subparsers.choices:
            # Arguments provided but no valid subcommand - insert 'run'
            sys.argv.insert(1, "run")
        return self._parser.parse_args()

    def execute(self, args: argparse.Namespace) -> None:
        """Main execution method that handles subcommands and legacy support."""

        # Handle legacy task listing
        if hasattr(args, "tasks") and args.tasks in [
            "list",
            "list_groups",
            "list_subtasks",
            "list_tags",
        ]:
            print(
                f"'--tasks {args.tasks}' is no longer supported.\n"
                f"Use the 'list' command instead:\n",
                file=sys.stderr,
            )

            # Show list command help
            list_parser = self._subparsers.choices["list"]
            list_parser.print_help()
            sys.exit(1)

        args.func(args)
