from typing import Any


DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_GEN_TOKS = 256
DEFAULT_RANDOM_SEED = 0
DEFAULT_OTHER_SEED = 1234


def default_gen_kwargs(
    until: str | list[str] | None, max_gen_toks: int = DEFAULT_MAX_GEN_TOKS
) -> dict[str, Any]:
    """Returns default generation kwargs for LM evaluation."""
    _gen = {
        "temperature": 0.0,
        "do_sample": False,
        "max_gen_toks": max_gen_toks,
    }
    if until is not None:
        _gen["until"] = [until] if isinstance(until, str) else until
    else:
        _gen["until"] = []
    return _gen
