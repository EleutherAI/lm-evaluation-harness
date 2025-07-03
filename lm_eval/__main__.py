from typing import Union
import argparse

from lm_eval._cli import CLIParser


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    """Main CLI entry point with subcommand and legacy support."""
    parser = CLIParser()

    if args is None:
        # Parse from command line
        parser.execute()
    else:
        # External call with pre-parsed args - use legacy mode
        parser._handle_legacy_mode(args)


if __name__ == "__main__":
    cli_evaluate()