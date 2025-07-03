import argparse
import json
import logging
from abc import ABC, abstractmethod
from typing import Union


def try_parse_json(value: str) -> Union[str, dict, None]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings."
            )
        return value


def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])

    return items


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
