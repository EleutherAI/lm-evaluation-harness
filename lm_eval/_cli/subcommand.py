import argparse
from abc import ABC, abstractmethod


class SubCommand(ABC):
    """Base class for all subcommands."""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def create(cls, subparsers: argparse._SubParsersAction):
        """Factory method to create and register a command instance."""
        return cls(subparsers)

    @abstractmethod
    def _add_args(self) -> None:
        """Add arguments specific to this subcommand."""
        pass
