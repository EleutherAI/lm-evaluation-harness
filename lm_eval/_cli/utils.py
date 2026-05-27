import argparse
import ast
import json
import logging
from collections.abc import Sequence
from typing import Any


eval_logger = logging.getLogger(__name__)


def try_parse_json(value: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
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
            ) from None
        return value


def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
) -> list[int | None]:
    """Parses a string of integers or 'None' values separated by a specified character into a list.

    Validates the number of items against specified minimum and maximum lengths and fills missing values with defaults.
    """

    def parse_value(item):
        """Parses an individual item, converting it to an integer or `None`."""
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise ValueError(f"{item} is not an integer or None") from None

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


def request_caching_arg_to_dict(cache_requests: str | None) -> dict[str, bool]:
    """Convert a request caching argument to a dictionary."""
    if cache_requests is None:
        return {}
    if cache_requests not in {"true", "refresh", "delete"}:
        raise argparse.ArgumentTypeError(
            f"invalid value '{cache_requests}' (choose from true, refresh, delete)"
        )
    return {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }


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


def handle_cli_value_string(arg: str) -> bool | int | float | str:
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        try:
            return ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            return arg


def key_val_to_dict(args: str) -> dict[str, Any]:
    """Parse model arguments from a string into a dictionary."""
    res = {}
    if not args:
        return res

    for k, v in (item.split("=", 1) for item in args.split(",")):
        v = handle_cli_value_string(v)
        if k in res:
            eval_logger.warning(f"Overwriting key '{k}': {res[k]!r} -> {v!r}")
        res[k] = v
    return res


class MergeDictAction(argparse.Action):
    """Argparse action that parses key=value args and merges them into a dict."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        current = vars(namespace).setdefault(self.dest, {}) or {}

        if not values:
            return

        # e.g. parses `{"pretrained":"/models/openai_gpt-oss-20b","dtype":"auto","chat_template_args":{"reasoning_effort":"low"},"enable_thinking": true,"think_end_token":"<|message|>"}`.
        result = try_parse_json(values[0])

        if isinstance(result, dict):
            current = {**current, **result}
        else:
            # e.g. parses `max_gen_toks=8000`
            if values:
                for v in values:
                    v = key_val_to_dict(v)
                    if overlap := current.keys() & v.keys():
                        eval_logger.warning(
                            rf"{option_string or self.dest}: Overwriting {', '.join(f'{k}: {current[k]!r} -> {v[k]!r}' for k in overlap)}"
                        )
                    current.update(v)

        setattr(namespace, self.dest, current)


class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        values = values or []
        assert values, f"--{self.dest} passed without any values"
        for v in values:
            items.extend(v.split(","))
        setattr(namespace, self.dest, items)
