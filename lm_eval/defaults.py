from typing import Any


def default_gen_kwargs(
    until: str | list[str] | None, max_gen_toks: int = 256
) -> dict[str, Any]:
    """Returns default generation kwargs for LM evaluation."""
    _gen = {
        "temperature": 0.0,
        "do_sample": False,
        "max_gen_toks": max_gen_toks,
    }
    if until is not None:
        _gen["until"] = [until] if isinstance(until, str) else until
    return _gen
