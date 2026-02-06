import os
from typing import Any


DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_GEN_TOKS = 256
DEFAULT_RANDOM_SEED = 0
DEFAULT_OTHER_SEED = 1234

# Environment variables


def _strtobool(val: str) -> bool:
    """Convert a string representation of truth to a bool."""
    _TRUTHY = {"1", "true", "yes", "on"}
    _FALSY = {"0", "false", "no", "off", ""}
    val = val.lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    raise ValueError(f"invalid truth value {val!r}")


def _envbool(var: str, default: bool = False) -> bool:
    """Read an environment variable as a bool."""
    val = os.environ.get(var)
    if val is None:
        return default
    return _strtobool(val)


LOGGING_LEVEL = os.environ.get("LMEVAL_LOG_LEVEL", "INFO")
LMEVAL_HASHMM = _envbool("LMEVAL_HASHMM", default=True)
DISABLE_MULTIPROC = _envbool("LMEVAL_DISABLE_MULTIPROC", default=False)


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
