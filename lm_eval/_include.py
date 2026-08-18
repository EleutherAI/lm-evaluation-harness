"""Runtime imports for user-supplied models/metrics/filters.

A .py file passed via `--include_path` is imported under a synthetic module
name. `sys.path` is not modified, so sibling imports from that file will
fail. Users with multi-file code should ship a package and use
`--include_module`.

Directories passed via `--include_path` are not touched here; they remain
YAML task-discovery roots.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import sys
from pathlib import Path


eval_logger = logging.getLogger(__name__)

_imported_paths: set[Path] = set()
_imported_modules: set[str] = set()


def _already_loaded(resolved: Path) -> bool:
    """Return True if a module whose __file__ equals resolved is in sys.modules."""
    for mod in list(sys.modules.values()):
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        try:
            if Path(mod_file).resolve() == resolved:
                return True
        except OSError:
            continue
    return False


def _import_file(path: Path) -> None:
    """Import the given .py file. Idempotent across repeated calls."""
    resolved = path.resolve()
    if resolved in _imported_paths:
        return
    if _already_loaded(resolved):
        _imported_paths.add(resolved)
        return

    suffix = hashlib.sha1(str(resolved).encode(), usedforsecurity=False).hexdigest()[:8]
    mod_name = f"lm_eval_user_{resolved.stem}_{suffix}"
    spec = importlib.util.spec_from_file_location(mod_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {resolved}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as err:
        raise ImportError(
            f"Failed to import user module {resolved}: {err}. "
            "Sibling .py imports are not supported; package the module "
            "and use --include_module."
        ) from err
    except Exception as err:
        raise ImportError(f"Failed to import user module {resolved}: {err}") from err

    _imported_paths.add(resolved)
    eval_logger.info("Imported user module from %s", resolved)


def task_discovery_path(include_path: str | Path | None) -> str | None:
    """Map an --include_path value to the YAML task-discovery root.

    For a .py file, returns its parent directory so YAML tasks next to
    the file are still discovered. Directories and None pass through.
    """
    if include_path is None:
        return None
    p = Path(include_path)
    if p.is_file() and p.suffix == ".py":
        return str(p.parent)
    return str(include_path)


def import_user_modules(
    include_path: str | Path | None = None,
    include_module: str | list[str] | None = None,
) -> None:
    """Import user-supplied Python so @register_* decorators run.

    include_path: only acted on when it points at a .py file. Other values
    (directories, non-.py files, missing paths) are skipped; TaskManager
    handles YAML discovery for directories.

    include_module: one or more dotted names passed to importlib.
    """
    if include_path:
        path = Path(include_path)
        if path.is_file() and path.suffix == ".py":
            _import_file(path)
        elif path.exists() and not path.is_dir():
            eval_logger.warning(
                "--include_path %s: not a .py file and not a directory; "
                "skipped for Python import.",
                path,
            )

    if include_module:
        names = (
            [include_module]
            if isinstance(include_module, str)
            else list(include_module)
        )
        for name in names:
            if name in _imported_modules:
                continue
            try:
                importlib.import_module(name)
            except Exception as err:
                raise ImportError(
                    f"Failed to import user module '{name}': {err}"
                ) from err
            _imported_modules.add(name)
            eval_logger.info("Imported user module '%s'", name)
