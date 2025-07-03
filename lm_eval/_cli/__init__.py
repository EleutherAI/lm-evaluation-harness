"""
CLI subcommands to run from terminal.
"""

from lm_eval._cli.base import SubCommand
from lm_eval._cli.eval import Eval
from lm_eval._cli.list import List
from lm_eval._cli.run import Run
from lm_eval._cli.validate import Validate


__all__ = [
    "SubCommand",
    "Run",
    "List",
    "Validate",
    "Eval",
]
