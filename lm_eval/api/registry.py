"""Registry system for lm_eval components.

This module provides a centralized registration system for models, tasks, metrics,
filters, and other components in the lm_eval framework.

## Usage Examples

### Registering a Model
```python
from lm_eval.api.registry import register_model
from lm_eval.api.model import LM

@register_model("my-model")
class MyModel(LM):
    def __init__(self, **kwargs):
        ...
```

### Registering with Lazy Loading
```python
# Register without importing the actual implementation
model_registry.register("lazy-model", target="my_package.models: LazyModel")
```

### Looking up Components
```python
from lm_eval.api.registry import get_model

# Get a model class
model_cls = get_model("gpt-j")
model = model_cls(**config)
```
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import logging
import threading
from collections.abc import Callable
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload


eval_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lm_eval.api.filter import Filter
    from lm_eval.api.metrics.metric import Metric
    from lm_eval.api.model import LM
    from lm_eval.scorers import Scorer

DEFAULT_METRIC_REGISTRY: Mapping[str, list[str]] = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


__all__ = [
    "DEFAULT_METRIC_REGISTRY",
    "Registry",
    "aggregation_registry",
    "filter_registry",
    "get_aggregation",
    "get_filter",
    "get_model",
    "get_reduction",
    "get_scorer",
    "metric_registry",
    "model_registry",
    "reduction_registry",
    "register_aggregation",
    "register_filter",
    "register_metric",
    "register_model",
    "register_reduction",
    "register_scorer",
    "scorer_registry",
]


_T = TypeVar("_T")
_D = TypeVar("_D")
_Fn = TypeVar("_Fn", bound=Callable)
Placeholder = str | md.EntryPoint

# Sentinel for distinguishing "no default" from "default=None"
_MISSING: Any = object()


@lru_cache(maxsize=16)
def _materialise_placeholder(ph: Placeholder) -> Any:
    """Materialize a lazy placeholder into the actual object.

    This is at module level to avoid memory leaks from lru_cache on instance methods.

    Args:
        ph: Either a string path "module:object" or an EntryPoint instance

    Returns:
        The loaded object

    Raises:
        ValueError: If the string format is invalid
        ImportError: If the module cannot be imported
        AttributeError: If the object doesn't exist in the module
    """
    if isinstance(ph, str):
        mod, _, attr = ph.partition(":")
        if not attr:
            raise ValueError(f"Invalid lazy path '{ph}', expected 'module:object'")
        return getattr(importlib.import_module(mod), attr)
    return ph.load()


def _suggest_similar(
    query: str, options: Iterable[str], max_suggestions: int = 3
) -> list[str]:
    """Return similar option names using prefix/substring matching."""
    query_lower = query.lower()
    suggestions = []
    for opt in options:
        opt_lower = opt.lower()
        if query_lower in opt_lower or (
            len(query_lower) >= 3 and opt_lower.startswith(query_lower[:3])
        ):
            suggestions.append(opt)
            if len(suggestions) >= max_suggestions:
                break
    return suggestions


def _build_key_error_msg(name: str, alias: str, keys: Iterable[str]) -> str:
    """Build a helpful KeyError message with suggestions."""
    suggestions = _suggest_similar(alias, keys)
    available = sorted(keys)

    msg = f"Unknown {name} '{alias}'."
    if suggestions:
        msg += f" Did you mean: {', '.join(suggestions)}?"
    msg += f"\nAvailable: {', '.join(available[:20])}"
    if len(available) > 20:
        msg += f"... ({len(available)} total)"
    return msg


class Registry(Generic[_T]):
    """Thread-safe dict mapping string aliases to objects or lazy placeholders.

    Lazy placeholders ("module.path:attr" strings or EntryPoints) are
    materialized on first access via `get()`. Optional `base_cls` enforces
    type constraints. Call `freeze()` to make read-only.
    """

    def __init__(
        self,
        name: str,
        *,
        base_cls: type[_T] | None = None,
        lazy_module: str | None = None,
    ) -> None:
        """Initialize a new registry.

        Args:
            name: Human-readable name for error messages (e.g., "model", "metric")
            base_cls: Optional base class that all registered objects must inherit from
            lazy_module: If set, auto-import this module on first ``get()`` when the
                registry is still empty.
        """
        self._name = name
        self._base_cls = base_cls
        self._lazy_module = lazy_module
        self._objs: dict[str, _T | Placeholder] = {}
        self._lock = threading.RLock()

    # Registration (decorator or direct call) --------------------------------------

    def register(
        self,
        *aliases: str,
        target: _T | Placeholder | None = None,
    ) -> Callable[[_T], _T]:
        """Register an object under one or more aliases.

        Can be used as a decorator or called directly for direct registration.

        Args:
            *aliases: Names to register the object under. If empty, uses object's __name__
            target: For direct calls only - a value, placeholder string "module:object",
                or EntryPoint to register under the alias

        Returns:
            Decorator function (or no-op if direct registration)

        Examples:
            >>> # As decorator
            >>> @model_registry.register("name1", "name2")
            >>> class MyModel(LM):
            ...     pass
            >>>
            >>> # Direct registration with a lazy placeholder
            >>> model_registry.register("lazy-name", target="mymodule:MyModel")

        Raises:
            ValueError: If alias is already registered with a different target
            TypeError: If an object doesn't inherit from base_cls (when specified)
        """

        def _store(alias: str, target: _T | Placeholder) -> None:
            current = self._objs.get(alias)
            # collision handling ------------------------------------------
            if current is not None and current != target:
                # allow placeholder → real object upgrade
                if (
                    isinstance(current, str)
                    and isinstance(target, type)
                    and current == f"{target.__module__}:{target.__name__}"
                ):
                    self._objs[alias] = target
                    return
                raise ValueError(
                    f"{self._name!r} alias '{alias}' already registered ("
                    f"existing={current}, new={target})"
                )
            # type check for concrete classes ----------------------------------------------
            if (
                self._base_cls is not None
                and isinstance(target, type)
                and not issubclass(target, self._base_cls)
            ):
                raise TypeError(
                    f"{target} must inherit from {self._base_cls} to be a {self._name}"
                )
            self._objs[alias] = target

        def decorator(obj: _T) -> _T:
            names = aliases or (getattr(obj, "__name__", str(obj)),)
            with self._lock:
                for name in names:
                    _store(name, obj)
            return obj

        # Direct call with target value
        if target is not None:
            if len(aliases) != 1:
                raise ValueError("Exactly one alias required when using 'target='")
            with self._lock:
                _store(aliases[0], target)
            # return no-op decorator for accidental use
            return lambda x: x

        return decorator

    # Lookup & materialisation --------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Auto-import ``lazy_module`` when the registry is still empty.

        Uses double-checked locking so only one thread triggers the import.
        """
        if self._lazy_module and len(self._objs) == 0:
            with self._lock:
                if len(self._objs) == 0:
                    importlib.import_module(self._lazy_module)

    def _materialise(self, ph: Placeholder) -> _T:
        """Materialize a placeholder using the module-level cached function.

        Args:
            ph: Placeholder to materialize

        Returns:
            The materialized object, cast to type T
        """
        return cast("_T", _materialise_placeholder(ph))

    @overload
    def get(self, alias: str) -> _T: ...

    @overload
    def get(self, alias: str, default: _D) -> _T | _D: ...

    def get(self, alias: str, default: _D | Any = _MISSING) -> _T | _D:
        """Retrieve an object by alias, materializing if needed.

        Thread-safe lazy loading: if the alias points to a placeholder,
        it will be loaded and cached before returning.  When ``lazy_module``
        is configured and the registry is still empty, the module is
        auto-imported first.

        Args:
            alias: The registered name to look up
            default: Default value to return if alias not found (can be None)

        Returns:
            The registered object, or default if not found

        Raises:
            KeyError: If an alias is not found and no default provided
            TypeError: If a materialized object doesn't match base_cls
            ImportError/AttributeError: If lazy loading fails
        """
        self._ensure_loaded()
        try:
            target = self._objs[alias]
        except KeyError as exc:
            if default is not _MISSING:
                return default
            raise KeyError(
                _build_key_error_msg(self._name, alias, self._objs.keys())
            ) from exc

        if isinstance(target, (str, md.EntryPoint)):
            with self._lock:
                # Re‑check under lock (another thread might have resolved it)
                fresh = self._objs[alias]
                if isinstance(fresh, (str, md.EntryPoint)):
                    concrete = self._materialise(fresh)
                    # Only update if not frozen (MappingProxyType)
                    if not isinstance(self._objs, MappingProxyType):
                        self._objs[alias] = concrete
                else:
                    concrete = fresh  # another thread did the job
            target = concrete

        # Late type/validator checks
        if (
            self._base_cls is not None
            and isinstance(target, type)
            and not issubclass(target, self._base_cls)
        ):
            raise TypeError(
                f"{target} does not inherit from {self._base_cls} (alias '{alias}')"
            ) from None
        return target

    def __getitem__(self, alias: str) -> _T:
        """Allow dict-style access: registry[alias]."""
        return self.get(alias)

    def __contains__(self, alias: str) -> bool:
        """Check if alias is registered."""
        self._ensure_loaded()
        return alias in self._objs

    def __iter__(self):
        """Iterate over registered aliases."""
        return iter(self._objs)

    def __len__(self):
        """Return the number of registered aliases."""
        return len(self._objs)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        materialized = sum(
            1 for v in self._objs.values() if not isinstance(v, (str, md.EntryPoint))
        )
        return f"Registry({self._name!r}, entries={len(self)}, materialized={materialized})"

    def keys(self):
        """Return all registered aliases."""
        return self._objs.keys()

    def values(self):
        """Return all registered objects.

        Note: Objects may be placeholders that haven't been materialized yet.
        """
        return self._objs.values()

    def items(self):
        """Return (alias, object) pairs.

        Note: Objects may be placeholders that haven't been materialized yet.
        """
        return self._objs.items()

    # Utilities -------------------------------------------------------------

    def origin(self, alias: str) -> str | None:
        """Get the source location of a registered object.

        Args:
            alias: The registered name

        Returns:
            "path/to/file.py:line_number" or None if not available
        """
        obj = self._objs.get(alias)
        if isinstance(obj, (str, md.EntryPoint)):
            return None
        try:
            path = inspect.getfile(obj)  # type: ignore[arg-type]
            line = inspect.getsourcelines(obj)[1]  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return None
        else:
            return f"{path}:{line}"

    def freeze(self):
        """Make the registry read-only to prevent further modifications.

        After freezing, attempts to register new objects will fail.
        This is useful for ensuring registry contents don't change after
        initialization.
        """
        with self._lock:
            self._objs = MappingProxyType(dict(self._objs))  # type: ignore[assignment]

    # Test helper --------------------------------
    def _clear(self):  # pragma: no cover
        """Erase registry (for isolated tests).

        Clears both the registry contents and the materialization cache.
        Only use this in test code to ensure a clean state between tests.
        """
        if isinstance(self._objs, MappingProxyType):
            self._objs = dict(self._objs)
        self._objs.clear()
        _materialise_placeholder.cache_clear()


# =============================================================================
# Deferred metric resolution
# =============================================================================


class _Deferred:
    """Lazy factory for Metric objects — resolved on first registry access."""

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory


class _MetricRegistry(Registry):
    """Registry that auto-resolves ``_Deferred`` entries on ``.get()``.

    Callers always receive a ``Metric`` (or the *default*) — never a raw
    ``_Deferred`` wrapper.
    """

    @overload
    def get(self, alias: str) -> Metric[Any, Any]: ...

    @overload
    def get(self, alias: str, default: _D) -> Metric[Any, Any] | _D: ...

    def get(self, alias, default=_MISSING):
        result = super().get(alias, default)
        if not isinstance(result, _Deferred):
            return result
        try:
            metric = result._factory()
        except Exception:
            eval_logger.error(
                "Failed to resolve deferred metric '%s'. This is a bug in "
                "the metric definition, not a missing metric.",
                alias,
                exc_info=True,
            )
            raise
        # Cache the resolved Metric for future lookups
        with self._lock:
            if not isinstance(self._objs, MappingProxyType):
                current = self._objs.get(alias)
                if isinstance(current, _Deferred):
                    self._objs[alias] = metric
        return metric


# =============================================================================
# Registry instances
# =============================================================================

_METRICS_MODULE = "lm_eval.api.metrics"
_REDUCE_MODULE = "lm_eval.api.metrics.reduce"

model_registry: Registry[type[LM]] = Registry("model")
filter_registry: Registry[type[Filter]] = Registry("filter")
scorer_registry: Registry = Registry("scorer", lazy_module="lm_eval.scorers")
aggregation_registry: Registry[Callable[..., float]] = Registry(
    "aggregation", lazy_module=_METRICS_MODULE
)
metric_registry: _MetricRegistry = _MetricRegistry(
    "metric", lazy_module=_METRICS_MODULE
)
reduction_registry: Registry[Callable] = Registry(
    "reduction", lazy_module=_REDUCE_MODULE
)


AGGREGATION_REGISTRY = aggregation_registry


# =============================================================================
# Model registration (using new Registry class)
# =============================================================================


def register_model(*names):
    """Decorator to register a model class.

    Args:
        *names: One or more names to register the model under

    Returns:
        Decorator function

    Example:
        >>> @register_model("my-model", "my-model-alias")
        >>> class MyModel(LM):
        ...     pass
    """
    # Import here to avoid circular import at module load time
    from lm_eval.api.model import LM

    def decorate(cls):
        assert issubclass(cls, LM), f"Model '{cls.__name__}' must extend LM class"
        # Use Registry's public API - it handles placeholder→concrete upgrades
        model_registry.register(*names)(cls)
        return cls

    return decorate


def get_model(model_name: str):
    """Get a model class by name.

    Args:
        model_name: The registered name of the model

    Returns:
        The model class

    Raises:
        ValueError: If model name is not found
    """
    # Auto-import models module if registry is empty (lazy initialization)
    if len(model_registry) == 0:
        import lm_eval.models  # noqa: F401

    try:
        return model_registry.get(model_name)
    except KeyError as e:
        raise ValueError(
            f"Attempted to load model '{model_name}', but no model for this name found! "
            f"Supported model names: {', '.join(model_registry.keys())}"
        ) from e


# Backward compatibility alias
MODEL_REGISTRY = model_registry


# =============================================================================
# Filter registration (using new Registry class)
# =============================================================================


def register_filter(name: str):
    """Decorator to register a filter class.

    Args:
        name: Name to register the filter under

    Returns:
        Decorator function
    """

    def decorate(cls):
        if name in filter_registry:
            eval_logger.info(f"Registering filter `{name}` that is already in Registry")
        # Use Registry's public API for registration
        filter_registry.register(name)(cls)
        return cls

    return decorate


def get_filter(filter_name: str | Callable) -> Callable:
    """Get a filter by name.

    Args:
        filter_name: The registered name of the filter, or a callable

    Returns:
        The filter class/function

    Raises:
        KeyError: If a filter name is not found and is not callable
    """
    if callable(filter_name):
        return filter_name
    try:
        return filter_registry.get(cast("str", filter_name))
    except KeyError:
        eval_logger.warning(f"filter `{filter_name}` is not registered!")
        raise


# Backward compatibility alias
FILTER_REGISTRY = filter_registry


# =============================================================================
# Scorer
# =============================================================================


def register_scorer(*names: str):
    """Decorator to register a scorer class.

    Args:
        *names: One or more names to register the scorer under

    Returns:
        Decorator function
    """

    def decorate(cls):
        for name in names:
            scorer_registry.register(name)(cls)
        return cls

    return decorate


def get_scorer(scorer_name: str) -> type[Scorer]:
    """Get a scorer class by name.

    Args:
        scorer_name: The registered name of the scorer

    Returns:
        The scorer class

    Raises:
        KeyError: If scorer name is not found
    """
    return scorer_registry.get(scorer_name)


# =============================================================================
# Metric registration (using new Registry class)
# =============================================================================


def register_metric(
    metric: str,
    *,
    higher_is_better: bool = True,
    aggregation: str | None = None,
    reduction: str | None = None,
    output_type: str | list[str] = "multiple_choice",
) -> Callable[[_Fn], _Fn]:
    """Decorator to register a function or class as a named ``Metric``.

    The metric is constructed lazily on first use, keeping import-time
    work minimal.

    Args:
        metric: Name to register the metric under
        higher_is_better: Whether higher scores are better (default True)
        aggregation: Name of aggregation function to use
        reduction: Name of reduction function to use
        output_type: str or list of output type names

    Returns:
        Decorator function
    """
    args: dict[str, Any] = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "output_type": output_type,
    }
    if aggregation is not None:
        args["aggregation"] = aggregation
    if reduction is not None:
        args["reduction"] = reduction

    def decorate(fn: _Fn) -> _Fn:
        name = metric

        def _build():
            from lm_eval.api.metrics.metric import Metric, take_first

            hib = args.get("higher_is_better", True)
            output_type = args.get("output_type", "multiple_choice")
            otp = [output_type] if isinstance(output_type, str) else list(output_type)

            agg_fn = None
            if "aggregation" in args:
                agg_fn = aggregation_registry.get(args["aggregation"])

            red_fn = take_first
            if "reduction" in args:
                red_fn = reduction_registry.get(args["reduction"])

            # CorpusMetric classes: detect via duck typing to avoid circular
            # import (corpus.py imports register_metric at module level).
            if (
                isinstance(fn, type)
                and hasattr(fn, "aggregation")
                and hasattr(fn, "reduce")
            ):
                instance = fn()
                return Metric(
                    name=name,
                    fn=cast("Any", instance),
                    aggregation=cast("Any", instance.aggregation),
                    reduction=cast("Any", instance.reduce),
                    higher_is_better=hib,
                    output_type=set(otp),
                )

            return Metric(
                name=name,
                fn=fn,
                aggregation=agg_fn,
                reduction=red_fn,
                higher_is_better=hib,
                output_type=set(otp),
            )

        try:
            metric_registry.register(name, target=_Deferred(_build))
        except ValueError:
            eval_logger.error(
                "Failed to register metric '%s'. This metric will NOT be available.",
                name,
                exc_info=True,
            )
            raise

        return fn

    return decorate


def register_aggregation(name: str):
    """Decorator to register an aggregation function.

    Args:
        name: Name to register the aggregation under

    Returns:
        Decorator function
    """

    def decorate(fn):
        aggregation_registry.register(name)(fn)
        return fn

    return decorate


def register_reduction(name: str):
    """Decorator to register a reduction function.

    Args:
        name: Name to register the reduction under

    Returns:
        Decorator function
    """

    def decorate(fn):
        reduction_registry.register(name)(fn)
        return fn

    return decorate


def get_reduction(name: str) -> Callable | None:
    """Get a reduction function by name.

    Args:
        name: The reduction name

    Returns:
        The reduction function, or None if not found
    """
    try:
        return reduction_registry.get(name)
    except KeyError:
        eval_logger.warning(f"{name} not a registered reduction!")
        return None


def get_aggregation(name: str) -> Callable[..., float] | None:
    """Get an aggregation function by name.

    Args:
        name: The aggregation name

    Returns:
        The aggregation function, or None if not found
    """
    try:
        return aggregation_registry.get(name)
    except KeyError:
        eval_logger.warning(f"{name} not a registered aggregation metric!")
        return None
