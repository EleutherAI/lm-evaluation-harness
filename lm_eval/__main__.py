from lm_eval._cli import CLIParser


def cli_evaluate() -> None:
    """Main CLI entry point with subcommand and legacy support."""
    parser = CLIParser()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    cli_evaluate()
