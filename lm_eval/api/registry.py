from __future__ import annotations

import importlib
import inspect
import threading
import warnings
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    cast,
)


try:  # Python≥3.10
    import importlib.metadata as md
except ImportError:  # pragma: no cover - fallback for 3.8/3.9 runtimes
    import importlib_metadata as md  # type: ignore

# Legacy exports (keep for one release, then drop)
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
    # legacy
    *LEGACY_EXPORTS,
]

T = TypeVar("T")


# ────────────────────────────────────────────────────────────────────────
# Generic Registry
# ────────────────────────────────────────────────────────────────────────


class Registry(Generic[T]):
    """Name -> object mapping with decorator helpers and **lazy import** support."""

    #: The underlying mutable mapping (might turn into MappingProxy on freeze)
    _objects: MutableMapping[str, T | str | md.EntryPoint]

    def __init__(
        self,
        name: str,
        *,
        base_cls: type[T] | None = None,
        store: MutableMapping[str, T | str | md.EntryPoint] | None = None,
        validator: Callable[[T], bool] | None = None,
    ) -> None:
        self._name: str = name
        self._base_cls: type[T] | None = base_cls
        self._objects = store if store is not None else {}
        self._metadata: dict[
            str, dict[str, Any]
        ] = {}  # Store metadata for each registered item
        self._validator = validator  # Custom validation function
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration helpers (decorator or direct call)
    # ------------------------------------------------------------------

    def _resolve_aliases(
        self, target: T | str | md.EntryPoint, aliases: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Resolve aliases for registration."""
        if not aliases:
            return (getattr(target, "__name__", str(target)),)
        return aliases

    def _check_and_store(
        self,
        alias: str,
        target: T | str | md.EntryPoint,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Check constraints and store the target with optional metadata.

        Collision policy:
        1. If alias doesn't exist → store it
        2. If identical value → silently succeed (idempotent)
        3. If lazy placeholder + matching concrete class → replace with concrete
        4. Otherwise → raise ValueError

        Type checking:
        - Eager for concrete classes at registration time
        - Deferred for lazy placeholders until materialization
        """
        with self._lock:
            # Case 1: New alias
            if alias not in self._objects:
                # Type check concrete classes before storing
                if self._base_cls is not None and isinstance(target, type):
                    if not issubclass(target, self._base_cls):  # type: ignore[arg-type]
                        raise TypeError(
                            f"{target} must inherit from {self._base_cls} "
                            f"to be registered as a {self._name}"
                        )
                self._objects[alias] = target
                if metadata:
                    self._metadata[alias] = metadata
                return

            existing = self._objects[alias]

            # Case 2: Identical value - idempotent
            if existing == target:
                return

            # Case 3: Lazy placeholder being replaced by its concrete class
            if isinstance(existing, str) and isinstance(target, type):
                mod_path, _, cls_name = existing.partition(":")
                if (
                    cls_name
                    and hasattr(target, "__module__")
                    and hasattr(target, "__name__")
                ):
                    expected_path = f"{target.__module__}:{target.__name__}"
                    if existing == expected_path:
                        self._objects[alias] = target
                        if metadata:
                            self._metadata[alias] = metadata
                        return

            # Case 4: Collision - different values
            raise ValueError(
                f"{self._name!r} '{alias}' already registered "
                f"(existing: {existing}, new: {target})"
            )

    def register(
        self,
        alias: str,
        target: T | str | md.EntryPoint,
        metadata: dict[str, Any] | None = None,
    ) -> T | str | md.EntryPoint:
        """Register a target (object or lazy placeholder) under the given alias.

        Args:
            alias: Name to register under
            target: Object to register (can be concrete object or lazy string "module:Class")
            metadata: Optional metadata to associate with this registration

        Returns:
            The target that was registered

        Examples:
            # Direct registration of concrete object
            registry.register("mymodel", MyModelClass)

            # Lazy registration with module path
            registry.register("mymodel", "mypackage.models:MyModelClass")
        """
        self._check_and_store(alias, target, metadata)
        return target

    def decorator(
        self,
        *aliases: str,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[T], T]:
        """Create a decorator for registering objects.

        Args:
            *aliases: Names to register under (if empty, uses object's __name__)
            metadata: Optional metadata to associate with this registration

        Returns:
            Decorator function that registers its target

        Example:
            @registry.decorator("mymodel", "model-v2")
            class MyModel:
                pass
        """

        def wrapper(obj: T) -> T:
            resolved_aliases = aliases or (getattr(obj, "__name__", str(obj)),)
            for alias in resolved_aliases:
                self.register(alias, obj, metadata)
            return obj

        return wrapper

    # ------------------------------------------------------------------
    # Lookup & materialisation
    # ------------------------------------------------------------------

    @lru_cache(maxsize=256)  # Bounded cache to prevent memory growth
    def _materialise(self, target: T | str | md.EntryPoint) -> T:
        """Import *target* if it is a dotted‑path string or EntryPoint."""
        if isinstance(target, str):
            mod, _, obj_name = target.partition(":")
            if not _:
                raise ValueError(
                    f"Lazy path '{target}' must be in 'module:object' form"
                )
            module = importlib.import_module(mod)
            return cast(T, getattr(module, obj_name))
        if isinstance(target, md.EntryPoint):
            return cast(T, target.load())
        return target  # concrete already

    def get(self, alias: str) -> T:
        # Fast path: check if already materialized without lock
        target = self._objects.get(alias)
        if target is not None and not isinstance(target, (str, md.EntryPoint)):
            # Already materialized and validated, return immediately
            return target

        # Slow path: acquire lock for materialization
        with self._lock:
            try:
                target = self._objects[alias]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown {self._name} '{alias}'. Available: "
                    f"{', '.join(self._objects)}"
                ) from exc

            # Double-check after acquiring a lock (may have been materialized by another thread)
            if not isinstance(target, (str, md.EntryPoint)):
                return target

            # Materialize the lazy placeholder
            concrete: T = self._materialise(target)

            # Swap placeholder with a concrete object (with race condition check)
            if concrete is not target:
                # Final check: another thread might have materialized while we were working
                current = self._objects.get(alias)
                if isinstance(current, (str, md.EntryPoint)):
                    # Still a placeholder, safe to replace
                    self._objects[alias] = concrete
                else:
                    # Another thread already materialized it, use their result
                    concrete = current  # type: ignore[assignment]

            # Late type check (for placeholders)
            if self._base_cls is not None and not issubclass(concrete, self._base_cls):  # type: ignore[arg-type]
                raise TypeError(
                    f"{concrete} does not inherit from {self._base_cls} "
                    f"(registered under alias '{alias}')"
                )

            # Custom validation - run on materialization
            if self._validator and not self._validator(concrete):
                raise ValueError(
                    f"{concrete} failed custom validation for {self._name} registry "
                    f"(registered under alias '{alias}')"
                )

            return concrete

    # Mapping / dunder helpers -------------------------------------------------

    def __getitem__(self, alias: str) -> T:  # noqa
        return self.get(alias)

    def __iter__(self):  # noqa
        return iter(self._objects)

    def __len__(self) -> int:  # noqa
        return len(self._objects)

    def items(self):  # noqa
        return self._objects.items()

    # Introspection -----------------------------------------------------------

    def origin(self, alias: str) -> str | None:
        obj = self._objects.get(alias)
        try:
            if isinstance(obj, str) or isinstance(obj, md.EntryPoint):
                return None  # placeholder - unknown until imported
            file = inspect.getfile(obj)  # type: ignore[arg-type]
            line = inspect.getsourcelines(obj)[1]  # type: ignore[arg-type]
            return f"{file}:{line}"
        except (
            TypeError,
            OSError,
            AttributeError,
        ):  # pragma: no cover - best-effort only
            # TypeError: object not suitable for inspect
            # OSError: file not found or accessible
            # AttributeError: object lacks expected attributes
            return None

    def get_metadata(self, alias: str) -> dict[str, Any] | None:
        """Get metadata for a registered item."""
        with self._lock:
            return self._metadata.get(alias)

    # Mutability --------------------------------------------------------------

    def freeze(self):
        """Make the registry *names* immutable (materialisation still works)."""
        with self._lock:
            if isinstance(self._objects, MappingProxyType):
                return  # already frozen
            self._objects = MappingProxyType(dict(self._objects))  # type: ignore[assignment]

    def clear(self):
        """Clear the registry (useful for tests). Cannot be called on frozen registries."""
        with self._lock:
            if isinstance(self._objects, MappingProxyType):
                raise RuntimeError("Cannot clear a frozen registry")
            self._objects.clear()
            self._metadata.clear()
            self._materialise.cache_clear()  # type: ignore[attr-defined]


# ────────────────────────────────────────────────────────────────────────
# Structured objects stored in registries
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    """Bundle compute fn, aggregator, and *higher‑is‑better* flag."""

    compute: Callable[[Any, Any], Any]
    aggregate: Callable[[Iterable[Any]], Mapping[str, float]]
    higher_is_better: bool = True
    output_type: str | None = None  # e.g., "probability", "string", "numeric"
    requires: list[str] | None = None  # Dependencies on other metrics/data


# ────────────────────────────────────────────────────────────────────────
# Concrete registries used by lm_eval
# ────────────────────────────────────────────────────────────────────────

from lm_eval.api.model import LM  # noqa: E402


model_registry: Registry[LM] = Registry("model", base_cls=LM)
task_registry: Registry[Callable[..., Any]] = Registry("task")
metric_registry: Registry[MetricSpec] = Registry("metric")
metric_agg_registry: Registry[Callable[[Iterable[Any]], Mapping[str, float]]] = (
    Registry("metric aggregation")
)
higher_is_better_registry: Registry[bool] = Registry("higher‑is‑better flag")
filter_registry: Registry[Callable] = Registry("filter")

# Default metric registry for output types
DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


def default_metrics_for(output_type: str) -> list[str]:
    """Get default metrics for a given output type dynamically.

    This walks the metric registry to find metrics that match the output type.
    Falls back to DEFAULT_METRIC_REGISTRY if no dynamic matches found.
    """
    # First, check static defaults
    if output_type in DEFAULT_METRIC_REGISTRY:
        return DEFAULT_METRIC_REGISTRY[output_type]

    # Walk metric registry for matching output types
    matching_metrics = []
    for name, metric_spec in metric_registry.items():
        if (
            isinstance(metric_spec, MetricSpec)
            and metric_spec.output_type == output_type
        ):
            matching_metrics.append(name)

    return matching_metrics if matching_metrics else []


# Aggregation registry - alias to the canonical registry for backward compatibility
AGGREGATION_REGISTRY = metric_agg_registry  # The registry itself is dict-like

# ────────────────────────────────────────────────────────────────────────
# Public helper aliases (legacy API)
# ────────────────────────────────────────────────────────────────────────

register_model = model_registry.decorator
get_model = model_registry.get

register_task = task_registry.decorator
get_task = task_registry.get


# Special handling for metric registration which uses different API
def register_metric(**kwargs):
    """Register a metric with metadata.

    Compatible with old registry API that used keyword arguments.
    """

    def decorate(fn):
        metric_name = kwargs.get("metric")
        if not metric_name:
            raise ValueError("metric name is required")

        # Determine aggregation function
        aggregate_fn: Callable[[Iterable[Any]], Mapping[str, float]] | None = None
        if "aggregation" in kwargs:
            agg_name = kwargs["aggregation"]
            try:
                aggregate_fn = metric_agg_registry.get(agg_name)
            except KeyError:
                raise ValueError(f"Unknown aggregation: {agg_name}")
        else:
            # No aggregation specified - use a function that raises NotImplementedError
            def not_implemented_agg(values):
                raise NotImplementedError(
                    f"No aggregation function specified for metric '{metric_name}'. "
                    "Please specify an 'aggregation' parameter."
                )

            aggregate_fn = not_implemented_agg

        # Create MetricSpec with the function and metadata
        spec = MetricSpec(
            compute=fn,
            aggregate=aggregate_fn,
            higher_is_better=kwargs.get("higher_is_better", True),
            output_type=kwargs.get("output_type"),
            requires=kwargs.get("requires"),
        )

        # Use a proper registry API with metadata
        metric_registry.register(metric_name, spec, metadata=kwargs)

        # Also register in higher_is_better registry if specified
        if "higher_is_better" in kwargs:
            higher_is_better_registry.register(metric_name, kwargs["higher_is_better"])

        return fn

    return decorate


def get_metric(name: str, hf_evaluate_metric=False):
    """Get a metric by name, with fallback to HF evaluate."""
    if not hf_evaluate_metric:
        try:
            spec = metric_registry.get(name)
            if isinstance(spec, MetricSpec):
                return spec.compute
            return spec
        except KeyError:
            import logging

            logging.getLogger(__name__).warning(
                f"Could not find registered metric '{name}' in lm-eval, searching in HF Evaluate library..."
            )

    # Fallback to HF evaluate
    try:
        import evaluate as hf_evaluate

        metric_object = hf_evaluate.load(name)
        return metric_object.compute
    except Exception:
        import logging

        logging.getLogger(__name__).error(
            f"{name} not found in the evaluate library! Please check https://huggingface.co/evaluate-metric",
        )
        return None


register_metric_aggregation = metric_agg_registry.decorator


def get_metric_aggregation(
    metric_name: str,
) -> Callable[[Iterable[Any]], Mapping[str, float]]:
    """Get the aggregation function for a metric."""
    # First, try to get from the metric registry (for metrics registered with aggregation)
    try:
        metric_spec = metric_registry.get(metric_name)
        if isinstance(metric_spec, MetricSpec) and metric_spec.aggregate:
            return metric_spec.aggregate
    except KeyError:
        pass  # Try the next registry

    # Fall back to metric_agg_registry (for standalone aggregations)
    try:
        return metric_agg_registry.get(metric_name)
    except KeyError:
        pass

    # If not found, raise an error
    raise KeyError(
        f"Unknown metric aggregation '{metric_name}'. Available: {list(metric_agg_registry)}"
    )


register_higher_is_better = higher_is_better_registry.decorator
is_higher_better = higher_is_better_registry.get

register_filter = filter_registry.decorator
get_filter = filter_registry.get


# Special handling for AGGREGATION_REGISTRY which works differently
def register_aggregation(name: str):
    """@deprecated Use metric_agg_registry.register() instead."""
    warnings.warn(
        "register_aggregation() is deprecated. Use metric_agg_registry.register() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def decorate(fn):
        # Use the canonical registry as a single source of truth
        if name in metric_agg_registry:
            raise ValueError(
                f"aggregation named '{name}' conflicts with existing registered aggregation!"
            )
        metric_agg_registry.register(name, fn)
        return fn

    return decorate


def get_aggregation(name: str) -> Callable[[Iterable[Any]], Mapping[str, float]] | None:
    """@deprecated Use metric_agg_registry.get() instead."""
    try:
        # Use the canonical registry
        return metric_agg_registry.get(name)
    except KeyError:
        import logging

        logging.getLogger(__name__).warning(
            f"{name} not a registered aggregation metric!"
        )
        return None


# ────────────────────────────────────────────────────────────────────────
# Optional PyPI entry‑point discovery - uncomment if desired
# ────────────────────────────────────────────────────────────────────────

# for _group, _reg in {
#     "lm_eval.models": model_registry,
#     "lm_eval.tasks": task_registry,
#     "lm_eval.metrics": metric_registry,
# }.items():
#     for _ep in md.entry_points(group=_group):
#         _reg.register(_ep.name, lazy=_ep)


# ────────────────────────────────────────────────────────────────────────
# Convenience
# ────────────────────────────────────────────────────────────────────────


def freeze_all() -> None:  # pragma: no cover
    """Freeze every global registry (idempotent)."""
    for _reg in (
        model_registry,
        task_registry,
        metric_registry,
        metric_agg_registry,
        higher_is_better_registry,
        filter_registry,
    ):
        _reg.freeze()


# ────────────────────────────────────────────────────────────────────────
# Backwards‑compatibility read‑only globals
# ────────────────────────────────────────────────────────────────────────

# These are direct aliases to the registries themselves, which already implement
# the Mapping protocol and provide read-only access to users (since _objects is private).
# This ensures they always reflect the current state of the registries, including
# items registered after module import.
#
# Note: We use type: ignore because Registry doesn't formally inherit from Mapping,
# but it implements all required methods (__getitem__, __iter__, __len__, items)

MODEL_REGISTRY: Mapping[str, LM] = model_registry  # type: ignore[assignment]
TASK_REGISTRY: Mapping[str, Callable[..., Any]] = task_registry  # type: ignore[assignment]
METRIC_REGISTRY: Mapping[str, MetricSpec] = metric_registry  # type: ignore[assignment]
METRIC_AGGREGATION_REGISTRY: Mapping[str, Callable] = metric_agg_registry  # type: ignore[assignment]
HIGHER_IS_BETTER_REGISTRY: Mapping[str, bool] = higher_is_better_registry  # type: ignore[assignment]
FILTER_REGISTRY: Mapping[str, Callable] = filter_registry  # type: ignore[assignment]
