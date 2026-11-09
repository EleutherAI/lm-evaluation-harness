import argparse
import sys
import textwrap

from lm_eval._cli.ls import List
from lm_eval._cli.run import Run
from lm_eval._cli.validate import Validate


class HarnessCLI:
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
                  lm-eval ls tasks

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
            # Backward compatibility: arguments provided but no valid subcommand - insert 'run'
            # TODO: add warning
            sys.argv.insert(1, "run")
        elif len(sys.argv) == 2 and "run" in sys.argv:
            # if only 'run' is specified, ensure it is treated as a subcommand
            self._subparsers.choices["run"].print_help()
            sys.exit(0)
        return self._parser.parse_args()

    def execute(self, args: argparse.Namespace) -> None:
        """Main execution method that handles subcommands and legacy support."""
        args.func(args)
