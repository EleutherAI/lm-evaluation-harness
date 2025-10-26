from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from functools import wraps
from inspect import getsource
from typing import Any, TypeVar

from frozendict import frozendict

from lm_eval import utils


T = TypeVar("T")


def serialize_callable(
    value: Callable[..., T] | str, keep_callable=False
) -> Callable[..., T] | str:
    """Serializes a given function or string.

    If 'keep_callable' is True, the original callable is returned.
    Otherwise, attempts to return the source code of the callable using 'getsource'.
    If serialization fails, returns the string representation.
    """
    if keep_callable:
        return value
    else:
        try:
            return getsource(value)  # type: ignore
        except (TypeError, OSError):
            return str(value)


def maybe_serialize(
    val: Callable[..., T] | Any, keep_callable=False
) -> Callable[..., T] | Any:
    """Conditionally serializes a value if it is callable."""

    return (
        serialize_callable(val, keep_callable=keep_callable) if callable(val) else val
    )


def create_choices(choices: list[str], choice_delimiter: str = "\n") -> list[str]:
    """Creates a question format from a list of choices."""
    return [f"{chr(65 + i)}" for i, _ in enumerate(choices)]


def create_mc_choices(choices: list[str], choice_delimiter: str = "\n") -> str:
    """Creates a multiple-choice question format from a list of choices."""
    formatted_choices = [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    return choice_delimiter.join(formatted_choices)


def create_cloze_choices(choices: list[str], choice_delimiter: str = "\n") -> str:
    """Creates a cloze-style question format from a list of choices."""


def doc_to_closure(fn: Callable[..., T]) -> Callable[..., T]:
    """Closure that allows the function to be called with 'self'."""

    @wraps(fn)
    def closure(self: Any, *args, **kwargs):
        return fn(*args, **kwargs)

    return closure


from frozendict import frozendict


def freezeargs(func):
    """Convert a mutable dictionary into immutable.
    Useful to be compatible with cache
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = (frozendict(arg) if isinstance(arg, dict) else arg for arg in args)
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


def process_field(
    doc: dict[str, str],
    field_spec: Any | None,
    *,
    digits: bool = False,
    lists: bool = False,
    default: Any | None = None,
) -> Any:
    """Processes a field from a document."""
    # fmt: off
    match field_spec:
        case None: return default
        case int(): return field_spec
        case func if callable(field_spec): return func(doc)
        case str() if field_spec in doc: return doc[field_spec]
    # fmt: on

    target_string = utils.apply_template(field_spec, doc)
    if lists:
        # TODO: fix sequence
        if isinstance(target_string, list) and any(
            x in ["{", "}", "(", ")", "[", "]"] for x in target_string
        ):
            return [utils.apply_template(x, doc) for x in target_string]
        return ast.literal_eval(target_string)
    elif digits:
        return int(target_string) if target_string.isdigit() else target_string

    return target_string or default


def merge_dicts(dict_list: list[dict[str, str]]) -> dict[str, Any]:
    """Merges a list of dictionaries into a single dictionary."""
    from collections import defaultdict

    result = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            result[key].append(value)

    return result
