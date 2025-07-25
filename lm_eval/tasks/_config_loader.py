from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


_Base = (
    yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.FullLoader
)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


# --------------------------------------------------------------------------- helpers
@functools.lru_cache(128)
def _mk_function_ctor(base_dir: Path, resolve: bool):
    def ctor(loader: yaml.Loader, node: yaml.Node):
        spec = loader.construct_scalar(node)  # type: ignore[arg-type]
        if not resolve:
            return str(base_dir.expanduser() / spec)
        return _import_func_in_yml(spec, base_dir)

    return ctor


@functools.lru_cache(maxsize=512)
def _make_loader(base_dir: Path, *, resolve_funcs: bool) -> type[yaml.Loader]:
    class Loader(_Base): ...  # type: ignore[no-redef]

    yaml.add_constructor(
        "!function",
        _mk_function_ctor(base_dir, resolve_funcs),
        Loader=Loader,
    )
    return Loader


@functools.lru_cache(maxsize=128)
def _import_func_in_yml(qual: str, base_dir: Path):
    """Import function from qual: utils.process_doc, checking local files first then standard imports.

    Args:
        qual: Qualified function name (e.g., 'utils.process_doc')
        base_dir: Directory to search for local modules
    """
    mod_path, _, fn_name = qual.rpartition(".")
    # 1) relative “utils.py” next to YAML
    rel = (base_dir / f"{mod_path.replace('.', '/')}.py").resolve()
    if rel.exists():
        mtime = rel.stat().st_mtime_ns
        key = f"{rel}:{mtime}"  # one module per mtime
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, rel)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {rel}") from None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            sys.modules[key] = mod
        return getattr(sys.modules[key], fn_name)

    # 2) already-importable module
    module = __import__(mod_path, fromlist=[fn_name])
    return getattr(module, fn_name)


@functools.lru_cache(maxsize=128)
def _import_fun_from_str(path_str: str) -> Any:
    """Import a function from a string in the form '/absolute/path/to/module.function_name'."""
    try:
        # Split off the function name from the rightmost dot
        module_path_str, function_name = path_str.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(
            f"Invalid path format: {path_str}. Expected format: /path/to/module.function_name"
        ) from e

    # Convert to Path and handle .py extension
    module_path = Path(module_path_str)
    if not module_path.suffix:
        module_path = module_path.with_suffix(".py")
    elif module_path.suffix != ".py":
        # If it has a non-.py suffix, the user might have included .py in the path
        # e.g., "/path/to/module.py.function_name"
        base_path = module_path.with_suffix("")
        if base_path.with_suffix(".py").exists():
            module_path = base_path.with_suffix(".py")

    if not module_path.exists():
        raise ImportError(f"Module file not found: {module_path}")

    # Use similar approach to _import_func_in_yml for consistency
    mtime = module_path.stat().st_mtime_ns
    cache_key = f"{module_path}:{mtime}"

    if cache_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(cache_key, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}") from None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[cache_key] = module

    module = sys.modules[cache_key]

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in module {module_path}"
        )

    return getattr(module, function_name)


def load_yaml(
    path: str | Path,
    *,
    resolve_functions: bool = True,
    resolve_includes: bool = True,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Pure data-loading helper.
    Returns a dict ready for higher-level interpretation.
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
    else:
        includes = cfg.pop("include")

    merged = {}
    for inc in includes if isinstance(includes, list) else [includes]:
        inc_path = (path.parent / inc) if not Path(inc).is_absolute() else Path(inc)
        merged.update(
            load_yaml(
                inc_path,
                resolve_functions=resolve_functions,
                resolve_includes=True,
                _seen=_seen,
            ),
        )
    merged.update(cfg)  # local keys win
    return merged
