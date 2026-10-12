from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib._bootstrap import _ModuleLockManager
from pathlib import Path
from threading import RLock
from typing import Any

import yaml


_Base = (
    yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.FullLoader
)
_IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
_MISSING = object()


@dataclass
class _PackageLoadState:
    modules: dict[str, Any]
    bindings: dict[str, dict[str, Any]]
    initializing: frozenset[str]
    active: int = 0
    succeeded: bool = False


_PACKAGE_STATES: dict[str, _PackageLoadState] = {}
_PACKAGE_STATE_LOCK = RLock()
_PACKAGE_REQUESTS: ContextVar[tuple[str, ...]] = ContextVar(
    "lm_eval_package_requests", default=()
)


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


def _exec_module(module_name: str, module_path: Path) -> Any:
    """Execute a file, publishing it before execution for circular imports."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}") from None
    module = importlib.util.module_from_spec(spec)
    module.__mtime__ = module_path.stat().st_mtime_ns  # type: ignore
    # Share Python's per-module lock and initialization flag with normal imports.
    with _ModuleLockManager(module_name):
        previous = sys.modules.get(module_name, _MISSING)
        parent_name, _, child = module_name.rpartition(".")
        parent = sys.modules.get(parent_name)
        binding = getattr(parent, child, _MISSING)
        spec._initializing = True
        sys.modules[module_name] = module
        if parent is not None:
            # An absent parent binding routes from-list imports through Python's
            # module lock; publishing it early would expose partial state.
            vars(parent).pop(child, None)
        try:
            # A changed main file can import helpers created since the finder
            # last scanned the package, even within one timestamp interval.
            _invalidate_package_caches(module.__package__)
            spec.loader.exec_module(module)
            if parent is not None:
                setattr(parent, child, module)
        except BaseException:
            if previous is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
            if parent is not None:
                if binding is _MISSING:
                    vars(parent).pop(child, None)
                else:
                    setattr(parent, child, binding)
            raise
        finally:
            spec._initializing = False
    return module


def _cached_module(module_name: str, module_path: Path) -> Any:
    """Reuse imports made by package initializers; reload only this file on change."""
    with _ModuleLockManager(module_name):
        existing = sys.modules.get(module_name)
        mtime = module_path.stat().st_mtime_ns
        if existing is not None and getattr(existing, "__mtime__", mtime) == mtime:
            existing.__mtime__ = mtime
            return existing
        return _exec_module(module_name, module_path)


def _module_parts(module_path: Path, root: Path) -> tuple[str, ...]:
    """Use filesystem spelling for Python names on case-insensitive Windows paths."""
    relative = module_path.relative_to(root)
    if os.name == "nt":
        actual = root
        for part in relative.parts:
            actual = next(
                child
                for child in actual.iterdir()
                if os.path.normcase(child.name) == os.path.normcase(part)
            )
        relative = actual.relative_to(root)
    return relative.with_suffix("").parts


def _package_name(root: Path) -> tuple[str, bool]:
    """Use an importable package's identity only when it resolves to this root."""
    root_name = root.name
    if os.name == "nt":
        root_name = next(child.name for child in root.parent.iterdir() if child == root)
    if root_name.isidentifier():
        try:
            spec = importlib.util.find_spec(root_name)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.origin and Path(spec.origin).resolve() == root / "__init__.py":
            return root_name, True
    # The directory name need not be an identifier, and must not shadow a library.
    identity = os.path.normcase(str(root))
    return "_lm_eval_external_" + hashlib.sha256(identity.encode()).hexdigest(), False


def _namespace_modules(package_name: str) -> dict[str, Any]:
    prefix = package_name + "."
    return {
        k: v
        for k, v in sys.modules.copy().items()
        if k == package_name or k.startswith(prefix)
    }


def _invalidate_package_caches(package_name: str) -> None:
    """Refresh package directory listings without global finder-cache deletion."""
    if not package_name:
        return
    root_name = package_name.partition(".")[0]
    paths = {
        path
        for module in _namespace_modules(root_name).values()
        for path in getattr(module, "__path__", ())
    }
    for path in paths:
        finder = sys.path_importer_cache.get(path)
        invalidate = getattr(finder, "invalidate_caches", None)
        if invalidate is not None:
            invalidate()


def _snapshot_package(package_name: str) -> _PackageLoadState:
    before = _namespace_modules(package_name)
    return _PackageLoadState(
        before,
        {k: vars(v).copy() for k, v in before.items() if v is not None},
        frozenset(
            k
            for k, v in before.items()
            if getattr(getattr(v, "__spec__", None), "_initializing", False)
        ),
    )


def _rollback_package(package_name: str, state: _PackageLoadState) -> None:
    current = _namespace_modules(package_name)
    # A preexisting standard initializer may have completed while this request
    # waited. Its successful dependency cache belongs to that import, not to
    # this failing YAML request; do not split those module identities.
    if any(
        current.get(name, _MISSING) is state.modules[name]
        and not getattr(
            getattr(state.modules[name], "__spec__", None), "_initializing", False
        )
        for name in state.initializing
    ):
        return
    # A snapshot can observe a standard import before its initializer finishes.
    # If it later failed and disappeared, never resurrect that partial object.
    restore = {
        k: v
        for k, v in state.modules.items()
        if k not in state.initializing or current.get(k, _MISSING) is v
    }
    # An unrelated standard import may still be executing in this namespace.
    # Do not remove its parent or dependencies out from under it.
    if any(
        getattr(getattr(v, "__spec__", None), "_initializing", False)
        for k, v in current.items()
        if state.modules.get(k, _MISSING) is not v
    ):
        return
    for name, module in current.items():
        parent_name, _, child = name.rpartition(".")
        parent = restore.get(parent_name)
        if name in state.initializing and current.get(name) is restore.get(
            name, _MISSING
        ):
            continue
        if parent is not None and vars(parent).get(child, _MISSING) is module:
            if child in state.bindings[parent_name]:
                setattr(parent, child, state.bindings[parent_name][child])
            else:
                vars(parent).pop(child, None)
        if name not in restore:
            sys.modules.pop(name, None)
    sys.modules.update(restore)


@contextmanager
def _package_transaction(package_name: str):
    """Roll back failed loads without undoing overlapping successful requests."""
    # Only bookkeeping holds this lock: never wait for imports or execute user
    # code under it, since standard module initialization may call this loader.
    with _PACKAGE_STATE_LOCK:
        state = _PACKAGE_STATES.get(package_name)
        if state is None:
            state = _snapshot_package(package_name)
            _PACKAGE_STATES[package_name] = state
        state.active += 1
    requests = _PACKAGE_REQUESTS.get()
    nested = package_name in requests
    token = _PACKAGE_REQUESTS.set((*requests, package_name))
    completed = False
    try:
        yield state
        completed = True
    finally:
        _PACKAGE_REQUESTS.reset(token)
        with _PACKAGE_STATE_LOCK:
            # Nested success belongs to its outer request: it cannot commit a
            # package whose enclosing initializer subsequently fails.
            state.succeeded |= completed and not nested
            state.active -= 1
            if not state.active:
                del _PACKAGE_STATES[package_name]
                if not state.succeeded:
                    _rollback_package(package_name, state)


def _load_package_module(root: Path, module_path: Path) -> Any:
    """Load an explicit package without sys.path edits, rolling back failed imports."""
    package_name, importable = _package_name(root)
    with _package_transaction(package_name):
        # Release the root lock before waiting for children: a child's standard
        # initializer can load another YAML while already holding its own lock.
        with _ModuleLockManager(package_name):
            if package_name not in sys.modules:
                if importable:
                    importlib.import_module(package_name)
                else:
                    _exec_module(package_name, root / "__init__.py")
        parts = _module_parts(module_path, root)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join((package_name, *parts))
        parent_name, _, _ = module_name.rpartition(".")
        if module_name not in sys.modules or parent_name not in sys.modules:
            # A dotted YAML path may name a newly created intermediate package.
            _invalidate_package_caches(package_name)
        if parent_name:
            importlib.import_module(parent_name)
        if (
            module_path.name != "__init__.py"
            and (module_path.with_suffix("") / "__init__.py").is_file()
        ):
            # YAML chooses the file even when Python would choose a same-named
            # directory package. Keep that package's normal identity intact.
            identity = os.path.normcase(str(module_path))
            alias = "_lm_eval_file_" + hashlib.sha256(identity.encode()).hexdigest()
            return _cached_module(parent_name + "." + alias, module_path)
        if module_name not in sys.modules:
            # Normal imports initialize intermediate packages and bind children.
            module = importlib.import_module(module_name)
            module.__mtime__ = module_path.stat().st_mtime_ns
            return module
        return _cached_module(module_name, module_path)


def _load_module_with_cache(module_path: Path) -> Any:
    """Load local functions with main-file mtime reload and normal helper caching."""
    module_path = module_path.resolve()
    tasks_dir = Path(__file__).resolve().parent
    if module_path.is_relative_to(tasks_dir):
        parts = _module_parts(module_path, tasks_dir)
        return _cached_module(".".join(("lm_eval", "tasks", *parts)), module_path)

    root = module_path.parent
    if (root / "__init__.py").is_file():
        while root.parent != root and (root.parent / "__init__.py").is_file():
            root = root.parent
        return _load_package_module(root, module_path)

    return _cached_module(
        os.path.normcase(str(module_path.with_suffix(""))), module_path
    )


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
        raise ValueError(f"Expected YAML dict from {path}, got {type(cfg).__name__}")  # noqa: TRY004 - preserve the YAML error contract

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
