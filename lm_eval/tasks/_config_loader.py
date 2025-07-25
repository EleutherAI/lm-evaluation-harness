from __future__ import annotations

import functools
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


_Base = yaml.CLoader if getattr(yaml, "__with_libyaml__", False) else yaml.FullLoader
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


# --------------------------------------------------------------------------- helpers
def _mk_function_ctor(base_dir: Path, resolve: bool):
    def ctor(loader: yaml.Loader, node: yaml.Node):
        spec = loader.construct_scalar(node)  # type: ignore[arg-type]
        if not resolve:
            return lambda *_, **__: None
        return _import_function(spec, base_dir)

    return ctor


@functools.lru_cache(maxsize=1024)
def _make_loader(base_dir: Path, *, resolve_funcs: bool) -> type[yaml.Loader]:
    class Loader(_Base): ...  # type: ignore[no-redef]

    yaml.add_constructor(
        "!function", _mk_function_ctor(base_dir, resolve_funcs), Loader=Loader
    )
    return Loader


@functools.lru_cache(maxsize=4096)
def _import_function(qual: str, base_dir: Path):
    mod_path, _, fn_name = qual.rpartition(".")
    # 1) relative “utils.py” next to YAML
    rel = (base_dir / f"{mod_path.replace('.', '/')}.py").resolve()
    if rel.exists():
        mtime = rel.stat().st_mtime_ns
        key = f"{rel}:{mtime}"  # one module per mtime
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, rel)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            sys.modules[key] = mod
        return getattr(sys.modules[key], fn_name)

    # 2) already‑importable module
    module = __import__(mod_path, fromlist=[fn_name])
    return getattr(module, fn_name)


# --------------------------------------------------------------------- public API
def load_yaml(
    path: str | Path,
    *,
    resolve_functions: bool = True,
    resolve_includes: bool = True,
    _seen: set[Path] | None = None,
) -> dict[str, str | Callable[..., Any]]:
    """
    Pure data‑loading helper.
    Returns a dict ready for higher‑level interpretation.
    •No task/group/tag semantics here.
    """
    path = Path(path).expanduser().resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ValueError(f"Include cycle at {path}")
    _seen.add(path)

    loader_cls = _make_loader(path.parent, resolve_funcs=resolve_functions)
    with path.open("rb") as fh:
        cfg = yaml.load(fh, Loader=loader_cls)

    if not resolve_includes or "include" not in cfg:
        return cfg

    merged = {}
    for inc in cfg.pop("include"):
        inc_path = (path.parent / inc) if not Path(inc).is_absolute() else Path(inc)
        merged.update(
            load_yaml(
                inc_path,
                resolve_functions=resolve_functions,
                resolve_includes=True,
                _seen=_seen,
            )
        )
    merged.update(cfg)  # local keys win
    return merged
