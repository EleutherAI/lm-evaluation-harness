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
    def _add_args(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments specific to this subcommand."""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the subcommand with the given arguments."""
        pass
