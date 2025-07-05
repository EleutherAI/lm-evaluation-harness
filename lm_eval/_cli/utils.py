import argparse
import json
import logging
from typing import Optional, Union


def try_parse_json(value: Union[str, dict, None]) -> Union[str, dict, None]:
    """Try to parse a string as JSON. If it fails, return the original string."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise ValueError(
                f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings."
            )
        return value


def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
) -> list[Union[int, None]]:
    """Parses a string of integers or 'None' values separated by a specified character into a list.
    Validates the number of items against specified minimum and maximum lengths and fills missing values with defaults."""

    def parse_value(item):
        """Parses an individual item, converting it to an integer or `None`."""
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise ValueError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise ValueError(
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


def request_caching_arg_to_dict(cache_requests: Optional[str]) -> dict[str, bool]:
    """Convert a request caching argument to a dictionary."""
    if cache_requests is None:
        return {}
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args


def check_argument_types(parser: argparse.ArgumentParser) -> None:
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        # Skip help, subcommands, and const actions
        if action.dest in ["help", "command"] or action.const is not None:
            continue
        if action.type is None:
            raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
        else:
            continue
