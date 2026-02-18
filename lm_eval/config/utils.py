import ast
from typing import Any, Literal

from typing_extensions import overload

from lm_eval import utils


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
