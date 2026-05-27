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
    from collections.abc import Callable, Iterable

    from lm_eval.api.filter import Filter
    from lm_eval.api.model import LM


__all__ = [
    # Core registry class
    "Registry",
    # Registry instances
    "model_registry",
    "filter_registry",
    "aggregation_registry",
    "metric_registry",
    "metric_agg_registry",
    "higher_is_better_registry",
    "freeze_all",
    # Helper functions
    "register_model",
    "get_model",
    "register_metric",
    "get_metric",
    "register_aggregation",
    "get_aggregation",
    "get_metric_aggregation",
    "is_higher_better",
    "register_filter",
    "get_filter",
    # Backward compat aliases (point to Registry instances)
    "MODEL_REGISTRY",
    "FILTER_REGISTRY",
    "METRIC_REGISTRY",
    "METRIC_AGGREGATION_REGISTRY",
    "AGGREGATION_REGISTRY",
    "HIGHER_IS_BETTER_REGISTRY",
    # Default metric configuration
    "DEFAULT_METRIC_REGISTRY",
]


T = TypeVar("T")
D = TypeVar("D")
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


class Registry(Generic[T]):
    """Thread-safe dict mapping string aliases to objects or lazy placeholders.

    Lazy placeholders ("module.path:attr" strings or EntryPoints) are
    materialized on first access via `get()`. Optional `base_cls` enforces
    type constraints. Call `freeze()` to make read-only.
    """

    def __init__(
        self,
        name: str,
        *,
        base_cls: type[T] | None = None,
    ) -> None:
        """Initialize a new registry.

        Args:
            name: Human-readable name for error messages (e.g., "model", "metric")
            base_cls: Optional base class that all registered objects must inherit from
        """
        self._name = name
        self._base_cls = base_cls
        self._objs: dict[str, T | Placeholder] = {}
        self._lock = threading.RLock()

    # Registration (decorator or direct call) --------------------------------------

    def register(
        self,
        *aliases: str,
        target: T | Placeholder | None = None,
    ) -> Callable[[T], T]:
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

        def _store(alias: str, target: T | Placeholder) -> None:
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

        def decorator(obj: T) -> T:  # type: ignore[valid-type]
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
                _store(aliases[0], target)  # type: ignore[arg-type]
            # return no-op decorator for accidental use
            return lambda x: x  # type: ignore[return-value]

        return decorator

    # Lookup & materialisation --------------------------------------------------

    def _materialise(self, ph: Placeholder) -> T:
        """Materialize a placeholder using the module-level cached function.

        Args:
            ph: Placeholder to materialize

        Returns:
            The materialized object, cast to type T
        """
        return cast("T", _materialise_placeholder(ph))

    @overload
    def get(self, alias: str) -> T: ...

    @overload
    def get(self, alias: str, default: D) -> T | D: ...

    def get(self, alias: str, default: D | Any = _MISSING) -> T | D:
        """Retrieve an object by alias, materializing if needed.

        Thread-safe lazy loading: if the alias points to a placeholder,
        it will be loaded and cached before returning.

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

    def __getitem__(self, alias: str) -> T:
        """Allow dict-style access: registry[alias]."""
        return self.get(alias)

    def __contains__(self, alias: str) -> bool:
        """Check if alias is registered."""
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
            return f"{path}:{line}"
        except Exception:  # pragma: no cover – best‑effort only
            return None

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
            self._objs = dict(self._objs)  # type: ignore[assignment]
        self._objs.clear()
        _materialise_placeholder.cache_clear()


# =============================================================================
# Registry instances
# =============================================================================

model_registry: Registry[type[LM]] = Registry("model")
filter_registry: Registry[type[Filter]] = Registry("filter")
aggregation_registry: Registry[Callable[..., float]] = Registry("aggregation")
metric_registry: Registry[Callable] = Registry("metric")
metric_agg_registry: Registry[Callable] = Registry("metric_aggregation")
higher_is_better_registry: Registry[bool] = Registry("higher_is_better")


def freeze_all():
    """Freeze all registries to prevent further modifications.

    This is useful for ensuring registry contents are immutable after
    initialization, preventing accidental modifications during runtime.
    """
    for r in (
        model_registry,
        filter_registry,
        aggregation_registry,
        metric_registry,
        metric_agg_registry,
        higher_is_better_registry,
    ):
        r.freeze()


# Backward compat aliases - these now point to Registry instances
METRIC_REGISTRY = metric_registry
METRIC_AGGREGATION_REGISTRY = metric_agg_registry
AGGREGATION_REGISTRY = aggregation_registry
HIGHER_IS_BETTER_REGISTRY = higher_is_better_registry

DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


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
    except KeyError as e:
        eval_logger.warning(f"filter `{filter_name}` is not registered!")
        raise e


# Backward compatibility alias
FILTER_REGISTRY = filter_registry


# =============================================================================
# Metric registration (using new Registry class)
# =============================================================================


def register_metric(**args):
    """Decorator to register a metric function.

    Args:
        **args: Keyword arguments including
            - metric: Name to register the metric under (required)
            - higher_is_better: Whether higher scores are better
            - aggregation: Name of aggregation function to use

    Returns:
        Decorator function
    """

    def decorate(fn):
        assert "metric" in args
        name = args["metric"]

        # Register the metric function
        metric_registry.register(name)(fn)

        # Register higher_is_better if provided
        if "higher_is_better" in args:
            higher_is_better_registry.register(name, target=args["higher_is_better"])

        # Register aggregation if provided
        if "aggregation" in args:
            agg_fn = aggregation_registry.get(args["aggregation"])
            metric_agg_registry.register(name, target=agg_fn)

        return fn

    return decorate


def get_metric(name: str, hf_evaluate_metric: bool = False) -> Callable | None:
    """Get a metric function by name.

    Args:
        name: The metric name
        hf_evaluate_metric: If True, skip the local registry and use HF evaluate

    Returns:
        The metric compute function, or None if not found
    """
    # Auto-import metrics module if registry is empty (lazy initialization)
    if len(metric_registry) == 0:
        import lm_eval.api.metrics  # noqa: F401

    if not hf_evaluate_metric:
        if name in metric_registry:
            return metric_registry.get(name)
        else:
            eval_logger.warning(
                f"Could not find registered metric '{name}' in lm-eval, searching in HF Evaluate library..."
            )

    try:
        import evaluate as hf_evaluate

        metric_object = hf_evaluate.load(name)
        return metric_object.compute
    except Exception:
        eval_logger.error(
            f"{name} not found in the evaluate library! Please check https://huggingface.co/evaluate-metric",
        )
        return None


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


def get_aggregation(name: str) -> Callable[..., float] | None:
    """Get an aggregation function by name.

    Args:
        name: The aggregation name

    Returns:
        The aggregation function, or None if not found
    """
    # Auto-import metrics module if registry is empty (lazy initialization)
    if len(aggregation_registry) == 0:
        import lm_eval.api.metrics  # noqa: F401

    try:
        return aggregation_registry.get(name)
    except KeyError:
        eval_logger.warning(f"{name} not a registered aggregation metric!")
        return None


def get_metric_aggregation(name: str) -> Callable[..., float] | None:
    """Get the aggregation function for a metric.

    Args:
        name: The metric name

    Returns:
        The aggregation function for that metric, or None if not found
    """
    # Auto-import metrics module if registry is empty (lazy initialization)
    if len(metric_agg_registry) == 0:
        import lm_eval.api.metrics  # noqa: F401

    try:
        return metric_agg_registry.get(name)
    except KeyError:
        eval_logger.warning(f"{name} metric is not assigned a default aggregation!")
        return None


def is_higher_better(metric_name: str) -> bool | None:
    """Check if higher values are better for a metric.

    Args:
        metric_name: The metric name

    Returns:
        True if higher is better, False otherwise, None if not found
    """
    # Auto-import metrics module if registry is empty (lazy initialization)
    if len(higher_is_better_registry) == 0:
        import lm_eval.api.metrics  # noqa: F401

    try:
        return higher_is_better_registry.get(metric_name)
    except KeyError:
        eval_logger.warning(
            f"higher_is_better not specified for metric '{metric_name}'!"
        )
        return None
