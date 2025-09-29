# from rich.traceback import install

from lm_eval._cli.harness import HarnessCLI
from lm_eval.utils import setup_logging


# install(show_locals=True)


def cli_evaluate() -> None:
    """Main CLI entry point."""
    setup_logging()
    parser = HarnessCLI()
    args = parser.parse_args()
    parser.execute(args)


if __name__ == "__main__":
    cli_evaluate()
