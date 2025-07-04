from lm_eval._cli.eval import Eval
from lm_eval.utils import setup_logging


def cli_evaluate() -> None:
    """Main CLI entry point with subcommand and legacy support."""
    setup_logging()
    parser = Eval()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    cli_evaluate()
