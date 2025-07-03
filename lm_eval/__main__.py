from lm_eval._cli.eval import Eval


def cli_evaluate() -> None:
    """Main CLI entry point with subcommand and legacy support."""
    parser = Eval()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    cli_evaluate()
