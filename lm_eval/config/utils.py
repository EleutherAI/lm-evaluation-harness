import ast
from collections.abc import Callable
from typing import Any, Literal

from typing_extensions import overload

from lm_eval import utils


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


@overload
def process_field(
    doc: Any,
    field_spec: Any,
    *,
    digits: Literal[False] = ...,
    lists: Literal[False] = ...,
) -> str | None: ...


@overload
def process_field(
    doc: Any,
    field_spec: Any,
    *,
    digits: Literal[True],
    lists: Literal[False] = ...,
) -> str | int | None: ...


@overload
def process_field(
    doc: Any,
    field_spec: Any,
    *,
    digits: Literal[False] = ...,
    lists: Literal[True],
) -> str | list[str] | None: ...


@overload
def process_field(
    doc: Any,
    field_spec: Any,
    *,
    digits: Literal[True],
    lists: Literal[True],
) -> str | int | list[str] | None: ...


def process_field(
    doc: dict[str, str],
    field_spec: Any | None,
    *,
    digits: bool = False,
    lists: bool = False,
) -> str | int | list[str] | None:
    """Processes a field from a document."""
    # fmt: off
    match field_spec:
        case None: return None
        case _ if callable(field_spec): return field_spec(doc)
        case int(): return field_spec
        case list(): return [utils.apply_template(x, doc) if isinstance(x, str) else x for x in field_spec]
        case str() if field_spec in doc: return doc[field_spec]
    # fmt: on

    target_string = utils.apply_template(field_spec, doc)

    if digits and isinstance(target_string, str) and target_string.isdigit():
        return int(target_string)

    if lists and isinstance(target_string, str):
        try:
            parsed = ast.literal_eval(target_string)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass

    return target_string
