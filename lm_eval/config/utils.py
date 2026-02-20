from __future__ import annotations

import functools
import re
import string
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from typing_extensions import overload


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_T = TypeVar("_T")


def serialize_callable(
    value: Callable[..., Any], *, keep_callable: bool = False
) -> Callable[..., Any] | str:
    """Serialize a callable to its source code string.

    If *keep_callable* is True the original callable is returned unchanged.
    Otherwise, we attempt ``inspect.getsource``; on failure we fall back to ``str()``.
    """
    from inspect import getsource

    if keep_callable:
        return value
    try:
        return getsource(value)
    except (TypeError, OSError):
        return str(value)


def _serialize_value(value: Any, keep_callable: bool) -> Any:
    """Recursively serialize callables in nested dicts/lists."""
    if callable(value):
        return serialize_callable(value, keep_callable=keep_callable)
    if isinstance(value, dict):
        return {k: _serialize_value(v, keep_callable) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item, keep_callable) for item in value]
    return value


def serialize_config(
    cfg, *, keep_callable: bool = False
) -> dict[str, str] | dict[str, Any]:
    """Convert a dataclass config to a plain dict, serializing callables.

    * ``None`` values are dropped.
    * Any callable value is serialized with :func:`serialize_callable`.
    """
    from dataclasses import asdict

    cfg_dict = asdict(cfg)
    for k, v in list(cfg_dict.items()):
        if v is None:
            cfg_dict.pop(k)
        else:
            cfg_dict[k] = _serialize_value(v, keep_callable)
    return cfg_dict


def regex_replace(text, pattern, repl, count: int = 0) -> str:
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, text, count=count)


def letter_choices(choices: Sequence[str], sep: str = "\n", style: str = ". ") -> str:
    r"""Format a list of choices with letter labels.

    Usage in Jinja: ``{{ choices | letter_choices }}``
    produces ``A. choice1\nB. choice2\n...``

    *style* controls the separator between the letter and the text
    (e.g. ``". "``, ``": "``, ``") "``).
    """
    return sep.join(
        f"{string.ascii_uppercase[i]}{style}{c}" for i, c in enumerate(choices)
    )


def answer_key_to_index(key: str | int, labels: Sequence[str] | None = None) -> int:
    """Convert an answer key to a 0-based index.

    Usage in Jinja: ``{{ answerKey | answer_key_to_index(choices.label) }}``

    Handles three cases:
    * *labels* provided → ``labels.index(key)`` (after stripping whitespace).
    * *key* is a letter (A-Z) → ordinal offset from ``'A'``.
    * *key* is a numeric string (1-based) → ``int(key) - 1``.
    """
    key_str = str(key).strip()
    if labels is not None:
        return list(labels).index(key_str)
    if len(key_str) == 1 and key_str.isalpha():
        return ord(key_str.upper()) - ord("A")
    return int(key_str) - 1


@functools.lru_cache(maxsize=1)
def _jinja_env():
    from jinja2 import BaseLoader, Environment, StrictUndefined

    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    env.filters["regex_replace"] = regex_replace
    env.filters["letter_choices"] = letter_choices
    env.filters["answer_key_to_index"] = answer_key_to_index
    return env


@functools.lru_cache(maxsize=256)
def _compile_tpl(src: str):
    return _jinja_env().from_string(src)


def apply_template(template: str, doc: dict, *args) -> str:
    try:
        return _compile_tpl(template).render(doc)
    except Exception as e:
        raise ValueError(
            f"Error rendering template: {template} with doc: {doc}, args: {args}"
        ) from e


@overload
def process_field(doc: dict[Any, Any], field_spec: None) -> None: ...
@overload
def process_field(doc: dict[Any, Any], field_spec: str) -> str: ...
@overload
def process_field(doc: dict[Any, Any], field_spec: int) -> int: ...
@overload
def process_field(doc: dict[Any, Any], field_spec: list[_T]) -> list[str | _T]: ...
@overload
def process_field(
    doc: dict[Any, Any], field_spec: Callable[[dict[Any, Any]], _T]
) -> _T: ...
def process_field(
    doc: dict[str, str],
    field_spec: Any | None,
) -> str | int | list | None:
    """Resolve a field spec against a document."""
    # fmt: off
    match field_spec:
        case None: return None
        case _ if callable(field_spec): return field_spec(doc)
        case int(): return field_spec
        case list(): return [apply_template(x, doc) if isinstance(x, str) else x for x in field_spec]
        case str() if field_spec in doc: return doc[field_spec]
    # fmt: on
    return apply_template(field_spec, doc)


@overload
def _coerce_list(value: None) -> None: ...
@overload
def _coerce_list(value: list[_T]) -> list[_T]: ...
@overload
def _coerce_list(value: str) -> str | list[str]: ...
@overload
def _coerce_list(value: int) -> int: ...
def _coerce_list(value):
    """Coerce a rendered field value: parse list literals to list."""
    import ast

    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return value


@overload
def _coerce_target(value: None, parse_list: bool = ...) -> None: ...
@overload
def _coerce_target(value: list[_T], parse_list: bool = ...) -> list[_T]: ...
@overload
def _coerce_target(value: str, parse_list: Literal[False] = ...) -> str | int: ...
@overload
def _coerce_target(value: str, parse_list: Literal[True]) -> str | int | list[str]: ...
@overload
def _coerce_target(value: int, parse_list: bool = ...) -> int: ...
def _coerce_target(value, parse_list=False):
    """Coerce a rendered field value: parse digit strings to int, list literals to list."""
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        if parse_list:
            return _coerce_list(value)
    return value


def _resolve_target_index(target, choices, doc) -> int | None:
    import logging

    eval_logger = logging.getLogger(__name__)
    if isinstance(target, (int, float)):
        idx = int(target)
        if idx < len(choices):
            return idx
        eval_logger.warning(
            f"Target index '{idx}' out of range for choices {choices} for doc:\n\n{doc}\n\n"
        )
        return None
    if isinstance(target, str):
        if target in choices:
            return choices.index(target)
        eval_logger.warning(
            f"Target '{target}' not found in choices {choices} for doc:\n\n{doc}\n\n"
        )
        return None
    return None
