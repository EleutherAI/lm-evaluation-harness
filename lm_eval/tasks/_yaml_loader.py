from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


_Base = (
    yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.FullLoader
)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def _mk_function_ctor(base_dir: Path, resolve: bool):
    def ctor(loader: yaml.Loader, node: yaml.Node):
        spec = loader.construct_scalar(node)  # type: ignore[arg-type]
        if not resolve:
            return str(base_dir.expanduser() / spec)
        return _import_func_in_yml(spec, base_dir)

    return ctor


def _make_loader(base_dir: Path, *, resolve_funcs: bool) -> type[yaml.Loader]:
    class Loader(_Base): ...  # type: ignore[no-redef]

    yaml.add_constructor(
        "!function",
        _mk_function_ctor(base_dir, resolve_funcs),
        Loader=Loader,
    )
    return Loader


def _load_module_with_cache(module_path: Path) -> Any:
    """Load a module from a file path with caching and hot-reload support.

    Args:
        module_path: Path to the Python file to load

    Returns:
        The loaded module
    """
    # Determine module name based on location
    path_str = str(module_path)

    # Check if this is a built-in task module
    if "/lm_eval/tasks/" in path_str:
        # Find the position of lm_eval/tasks/ in the path
        tasks_idx = path_str.find("/lm_eval/tasks/")
        if tasks_idx != -1:
            # Extract path starting from lm_eval/tasks/
            # e.g., /path/to/lm_eval/tasks/hellaswag/utils.py → hellaswag/utils.py
            relative_path = path_str[tasks_idx + len("/lm_eval/tasks/") :]
            # Remove .py and convert to module name
            # e.g., hellaswag/utils.py → lm_eval.tasks.hellaswag.utils
            module_parts = relative_path.replace(".py", "").replace("/", ".")
            module_name = f"lm_eval.tasks.{module_parts}"
        else:
            # Fallback to a full path if a pattern not found
            module_name = str(module_path.with_suffix(""))
    else:
        # External module - use a full path without extension
        module_name = str(module_path.with_suffix(""))

    # Check if we need to reload the module
    if module_name in sys.modules:
        existing_module = sys.modules[module_name]
        # Check if it was modified
        current_mtime = module_path.stat().st_mtime_ns
        if (
            hasattr(existing_module, "__mtime__")
            and existing_module.__mtime__ == current_mtime
        ):
            # Module hasn't changed, reuse it
            return existing_module

    # Load or reload the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}") from None
    module = importlib.util.module_from_spec(spec)
    # Store mtime for future checks
    module.__mtime__ = module_path.stat().st_mtime_ns  # type: ignore
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    sys.modules[module_name] = module
    return module


def _import_func_in_yml(qual: str, base_dir: Path):
    """Import function from qual: utils.process_doc, checking local files first then standard imports.

    Args:
        qual: Qualified function name (e.g., 'utils.process_doc')
        base_dir: Directory to search for local modules
    """
    mod_path, _, fn_name = qual.rpartition(".")
    if not mod_path:
        raise ValueError(
            f"Invalid function reference '{qual}': no module path (from YAML in {base_dir})"
        )
    # 1) relative "utils.py" next to YAML
    rel = (base_dir / f"{mod_path.replace('.', '/')}.py").resolve()
    if rel.exists():
        try:
            module = _load_module_with_cache(rel)
            return getattr(module, fn_name)
        except AttributeError as e:
            raise AttributeError(
                f"Module '{rel}' has no function '{fn_name}' (from YAML in {base_dir})"
            ) from e

    # 2) already-importable module
    try:
        module = __import__(mod_path, fromlist=[fn_name])
        return getattr(module, fn_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{mod_path}' for function '{fn_name}' (from YAML in {base_dir})"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Module '{mod_path}' has no function '{fn_name}' (from YAML in {base_dir})"
        ) from e


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

    module = _load_module_with_cache(module_path)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Function '{function_name}' not found in module {module_path}"
        )

    return getattr(module, function_name)


def load_yaml(
    path: str | Path,
    *,
    resolve_func: bool = True,
    recursive: bool = True,
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

    loader_cls = _make_loader(path.parent, resolve_funcs=resolve_func)
    with path.open("rb") as fh:
        # we don't use yaml.safe_load here because we want to support !function tags
        cfg = yaml.load(fh, Loader=loader_cls)  # noqa: S506

    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML dict from {path}, got {type(cfg).__name__}")

    if not recursive or "include" not in cfg:
        return cfg
    else:
        includes = cfg.pop("include")

    merged = {}
    for inc in includes if isinstance(includes, list) else [includes]:
        inc_path = (path.parent / inc) if not Path(inc).is_absolute() else Path(inc)
        inc_cfg = load_yaml(
            inc_path,
            resolve_func=resolve_func,
            recursive=True,
            _seen=_seen,
        )
        # Don't inherit task_list - it defines tasks for the included file only
        inc_cfg.pop("task_list", None)
        merged.update(inc_cfg)
    merged.update(cfg)  # local keys win
    return merged
