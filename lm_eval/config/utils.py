from __future__ import annotations

from inspect import getsource
from typing import Any, Callable


def serialize_callable(
    value: Callable[..., Any] | str, keep_callable=False
) -> Callable[..., Any] | str:
    """Serializes a given function or string.

    If 'keep_callable' is True, the original callable is returned.
    Otherwise, attempts to return the source code of the callable using 'getsource'.
    If serialization fails, returns the string representation.
    """
    if keep_callable:
        return value
    else:
        try:
            return getsource(value)
        except (TypeError, OSError):
            return str(value)


def maybe_serialize(val: Callable | Any, keep_callable=False) -> Callable | Any:
    """Conditionally serializes a value if it is callable."""

    return (
        serialize_callable(val, keep_callable=keep_callable) if callable(val) else val
    )
