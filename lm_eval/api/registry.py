"""Registry system for lm_eval components.

This module provides a centralized registration system for models, tasks, metrics,
filters, and other components in the lm_eval framework. The registry supports:

- Lazy loading with placeholders to improve startup time
- Type checking and validation
- Thread-safe registration and lookup
- Plugin discovery via entry points
- Backwards compatibility with legacy registration patterns

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

### Registering a Metric
```python
from lm_eval.api.registry import register_metric

@register_metric(
    metric="my_accuracy",
    aggregation="mean",
    higher_is_better=True
)
def my_accuracy_fn(items):
    ...
```

### Registering with Lazy Loading
```python
# Register without importing the actual implementation
model_registry.register("lazy-model", lazy="my_package.models:LazyModel")
```

### Looking up Components
```python
from lm_eval.api.registry import get_model, get_metric

# Get a model class
model_cls = get_model("gpt-j")
model = model_cls(**config)

# Get a metric function
metric_fn = get_metric("accuracy")
```
"""

from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Generic, TypeVar, Union, cast

from lm_eval.api.filter import Filter


try:
    import importlib.metadata as md  # Python ≥3.10
except ImportError:  # pragma: no cover – fallback for 3.8/3.9
    import importlib_metadata as md  # type: ignore

LEGACY_EXPORTS = [
    "DEFAULT_METRIC_REGISTRY",
    "AGGREGATION_REGISTRY",
    "register_model",
    "get_model",
    "register_task",
    "get_task",
    "register_metric",
    "get_metric",
    "register_metric_aggregation",
    "get_metric_aggregation",
    "register_higher_is_better",
    "is_higher_better",
    "register_filter",
    "get_filter",
    "register_aggregation",
    "get_aggregation",
    "MODEL_REGISTRY",
    "TASK_REGISTRY",
    "METRIC_REGISTRY",
    "METRIC_AGGREGATION_REGISTRY",
    "HIGHER_IS_BETTER_REGISTRY",
    "FILTER_REGISTRY",
]

__all__ = [
    # canonical
    "Registry",
    "MetricSpec",
    "model_registry",
    "task_registry",
    "metric_registry",
    "metric_agg_registry",
    "higher_is_better_registry",
    "filter_registry",
    "freeze_all",
    *LEGACY_EXPORTS,
]  # type: ignore

T = TypeVar("T")
Placeholder = Union[str, md.EntryPoint]


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


# Metric-specific metadata storage --------------------------------------------

_metric_meta: dict[str, dict[str, Any]] = {}


class Registry(Generic[T]):
    """A thread-safe registry for named objects with lazy loading support.

    The Registry provides a central location for registering and retrieving
    components by name. It supports:

    - Direct registration of objects
    - Lazy registration with placeholders (strings or entry points)
    - Type checking against a base class
    - Thread-safe operations
    - Freezing to prevent further modifications

    Example:
        >>> from lm_eval.api.model import LM
        >>> registry = Registry("models", base_cls=LM)
        >>>
        >>> # Direct registration
        >>> @registry.register("my-model")
        >>> class MyModel(LM):
        ...     pass
        >>>
        >>> # Lazy registration
        >>> registry.register("lazy-model", lazy="mypackage:LazyModel")
        >>>
        >>> # Retrieval (triggers lazy loading if needed)
        >>> model_cls = registry.get("my-model")
        >>> model = model_cls()
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
        lazy: T | Placeholder | None = None,
    ) -> Callable[[T], T]:
        """Register an object under one or more aliases.

        Can be used as a decorator or called directly for lazy registration.

        Args:
            *aliases: Names to register the object under. If empty, uses object's __name__
            lazy: For direct calls only - a placeholder string "module:object" or EntryPoint

        Returns:
            Decorator function (or no-op if lazy registration)

        Examples:
            >>> # As decorator
            >>> @model_registry.register("name1", "name2")
            >>> class MyModel(LM):
            ...     pass
            >>>
            >>> # Direct lazy registration
            >>> model_registry.register("lazy-name", lazy="mymodule:MyModel")

        Raises:
            ValueError: If alias is already registered with a different target
            TypeError: If an object doesn't inherit from base_cls (when specified)
        """

        def _store(alias: str, target: T | Placeholder) -> None:
            current = self._objs.get(alias)
            # collision handling ------------------------------------------
            if current is not None and current != target:
                # allow placeholder → real object upgrade
                # mod, _, cls = current.partition(":")
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

        # Direct call with *lazy* placeholder
        if lazy is not None:
            if len(aliases) != 1:
                raise ValueError("Exactly one alias required when using 'lazy='")
            with self._lock:
                _store(aliases[0], lazy)  # type: ignore[arg-type]
            # return no‑op decorator for accidental use
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
        return cast(T, _materialise_placeholder(ph))

    def get(self, alias: str) -> T:
        """Retrieve an object by alias, materializing if needed.

        Thread-safe lazy loading: if the alias points to a placeholder,
        it will be loaded and cached before returning.

        Args:
            alias: The registered name to look up

        Returns:
            The registered object

        Raises:
            KeyError: If alias not found
            TypeError: If materialized object doesn't match base_cls
            ImportError/AttributeError: If lazy loading fails
        """
        try:
            target = self._objs[alias]
        except KeyError as exc:
            raise KeyError(
                f"Unknown {self._name} '{alias}'. Available: {', '.join(self._objs)}"
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
        if self._base_cls is not None and not issubclass(target, self._base_cls):  # type: ignore[arg-type]
            raise TypeError(
                f"{target} does not inherit from {self._base_cls} (alias '{alias}')"
            )
        return target

    def __getitem__(self, alias: str) -> T:
        """Allow dict-style access: registry[alias]."""
        return self.get(alias)

    def __iter__(self):
        """Iterate over registered aliases."""
        return iter(self._objs)

    def __len__(self):
        """Return number of registered aliases."""
        return len(self._objs)

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
        Only use this in test code to ensure clean state between tests.
        """
        self._objs.clear()
        _materialise_placeholder.cache_clear()


# Structured object for metrics ------------------


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a metric including computation and aggregation functions.

    Attributes:
        compute: Function to compute metric on individual items
        aggregate: Function to aggregate multiple metric values into a single score
        higher_is_better: Whether higher values indicate better performance
        output_type: Optional type hint for the output (e.g., "generate_until" for perplexity)
        requires: Optional list of other metrics this one depends on
    """

    compute: Callable[[Any, Any], Any]
    aggregate: Callable[[Sequence[float]], float]
    higher_is_better: bool = True
    output_type: str | None = None
    requires: list[str] | None = None


# Canonical registries aliases ---------------------

from lm_eval.api.model import LM  # noqa: E402


model_registry = cast(Registry[type[LM]], Registry("model", base_cls=LM))
task_registry: Registry[Callable[..., Any]] = Registry("task")
metric_registry: Registry[MetricSpec] = Registry("metric")
metric_agg_registry: Registry[Callable[..., float]] = Registry("metric aggregation")
higher_is_better_registry: Registry[bool] = Registry("higher‑is‑better flag")
filter_registry: Registry[type[Filter]] = Registry("filter")

# Public helper aliases ------------------------------------------------------

register_model = model_registry.register
get_model = model_registry.get

register_task = task_registry.register
get_task = task_registry.get

register_filter = filter_registry.register
get_filter = filter_registry.get

# Metric helpers need thin wrappers to build MetricSpec ----------------------


def _no_aggregation_fn(values: Iterable[Any]) -> float:
    """Default aggregation that raises NotImplementedError.

    Args:
        values: Metric values to aggregate (unused)

    Raises:
        NotImplementedError: Always - this is a placeholder for metrics
                           that haven't specified an aggregation function
    """
    raise NotImplementedError(
        "No aggregation function specified for this metric. "
        "Please specify 'aggregation' parameter in @register_metric."
    )


def register_metric(**kw):
    """Decorator for registering metric functions.

    Creates a MetricSpec from the decorated function and keyword arguments,
    then registers it in the metric registry.

    Args:
        **kw: Keyword arguments including
            - metric: Name to register the metric under (required)
            - aggregation: Name of aggregation function in metric_agg_registry
            - higher_is_better: Whether higher scores are better (default: True)
            - output_type: Optional output type hint
            - requires: Optional list of required metrics

    Returns:
        Decorator function that registers the metric

    Example:
        >>> @register_metric(
        ...     metric="my_accuracy",
        ...     aggregation="mean",
        ...     higher_is_better=True
        ... )
        ... def compute_accuracy(items):
        ...     return sum(item["correct"] for item in items) / len(items)
    """
    name = kw["metric"]

    def deco(fn):
        spec = MetricSpec(
            compute=fn,
            aggregate=(
                metric_agg_registry.get(kw["aggregation"])
                if "aggregation" in kw
                else _no_aggregation_fn
            ),
            higher_is_better=kw.get("higher_is_better", True),
            output_type=kw.get("output_type"),
            requires=kw.get("requires"),
        )
        metric_registry.register(name, lazy=spec)
        _metric_meta[name] = kw
        higher_is_better_registry.register(name, lazy=spec.higher_is_better)
        return fn

    return deco


def get_metric(name, hf_evaluate_metric=False):
    """Get a metric compute function by name.

    First checks the local metric registry, then optionally falls back
    to HuggingFace evaluate library.

    Args:
        name: Metric name to retrieve
        hf_evaluate_metric: If True, suppress warning when falling back to HF

    Returns:
        The metric's compute function

    Raises:
        KeyError: If a metric is not found in registry or HF evaluate
    """
    try:
        spec = metric_registry.get(name)
        return spec.compute  # type: ignore[attr-defined]
    except KeyError:
        if not hf_evaluate_metric:
            import logging

            logging.getLogger(__name__).warning(
                f"Metric '{name}' not in registry; trying HF evaluate…"
            )
        try:
            import evaluate as hf

            return hf.load(name).compute  # type: ignore[attr-defined]
        except Exception:
            raise KeyError(f"Metric '{name}' not found anywhere") from None


register_metric_aggregation = metric_agg_registry.register
get_metric_aggregation = metric_agg_registry.get

register_higher_is_better = higher_is_better_registry.register
is_higher_better = higher_is_better_registry.get

# Legacy compatibility
register_aggregation = metric_agg_registry.register
get_aggregation = metric_agg_registry.get
DEFAULT_METRIC_REGISTRY = metric_registry
AGGREGATION_REGISTRY = metric_agg_registry


def freeze_all():
    """Freeze all registries to prevent further modifications.

    This is useful for ensuring registry contents are immutable after
    initialization, preventing accidental modifications during runtime.
    """
    for r in (
        model_registry,
        task_registry,
        metric_registry,
        metric_agg_registry,
        higher_is_better_registry,
        filter_registry,
    ):
        r.freeze()


# Backwards‑compat aliases ----------------------------------------

MODEL_REGISTRY = model_registry
TASK_REGISTRY = task_registry
METRIC_REGISTRY = metric_registry
METRIC_AGGREGATION_REGISTRY = metric_agg_registry
HIGHER_IS_BETTER_REGISTRY = higher_is_better_registry
FILTER_REGISTRY = filter_registry
