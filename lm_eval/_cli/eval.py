import argparse
import sys
import textwrap

from lm_eval._cli.list import List
from lm_eval._cli.run import Run
from lm_eval._cli.validate import Validate


class Eval:
    """Main CLI parser that manages all subcommands."""

    def __init__(self):
        self._parser = argparse.ArgumentParser(
            prog="lm-eval",
            description="Language Model Evaluation Harness",
            epilog=textwrap.dedent("""
                quick start:
                  # Basic evaluation
                  lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag

                  # List available tasks
                  lm-eval list tasks

                  # Validate task configurations
                  lm-eval validate --tasks hellaswag,arc_easy

                legacy compatibility:
                  The harness maintains backward compatibility with the original interface.
                  If no command is specified, 'run' is automatically inserted:

                  lm-eval --model hf --tasks hellaswag  # Equivalent to 'lm-eval run --model hf --tasks hellaswag'

                For documentation, visit: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._parser.set_defaults(func=lambda args: self._parser.print_help())
        self._subparsers = self._parser.add_subparsers(
            dest="command", help="Available commands", metavar="COMMAND"
        )
        Run.create(self._subparsers)
        List.create(self._subparsers)
        Validate.create(self._subparsers)

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
