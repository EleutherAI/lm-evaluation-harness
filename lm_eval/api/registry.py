from __future__ import annotations

import importlib
import inspect
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Generic, Type, TypeVar, Union, cast


try:
    import importlib.metadata as md  # Python ≥3.10
except ImportError:  # pragma: no cover – fallback for 3.8/3.9
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
Placeholder = Union[str, md.EntryPoint]  # light‑weight lazy token


# ────────────────────────────────────────────────────────────────────────
# Generic Registry
# ────────────────────────────────────────────────────────────────────────


class Registry(Generic[T]):
    """Name → object registry with optional lazy placeholders."""

    def __init__(
        self,
        name: str,
        *,
        base_cls: Union[Type[T], None] = None,
    ) -> None:
        self._name = name
        self._base_cls = base_cls
        self._objs: dict[str, Union[T, Placeholder]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration (decorator or direct call)
    # ------------------------------------------------------------------

    def register(
        self,
        *aliases: str,
        lazy: Union[T, Placeholder, None] = None,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[T], T]:
        """``@reg.register('foo')`` or ``reg.register('foo', lazy='pkg.mod:Obj')``."""

        def _store(alias: str, target: Union[T, Placeholder]) -> None:
            current = self._objs.get(alias)
            # ─── collision handling ────────────────────────────────────
            if current is not None and current != target:
                # allow placeholder → real object upgrade
                if isinstance(current, str) and isinstance(target, type):
                    mod, _, cls = current.partition(":")
                    if current == f"{target.__module__}:{target.__name__}":
                        self._objs[alias] = target
                        self._meta[alias] = metadata or {}
                        return
                raise ValueError(
                    f"{self._name!r} alias '{alias}' already registered ("  # noqa: B950
                    f"existing={current}, new={target})"
                )
            # ─── type check for concrete classes ───────────────────────
            if self._base_cls is not None and isinstance(target, type):
                if not issubclass(target, self._base_cls):  # type: ignore[arg-type]
                    raise TypeError(
                        f"{target} must inherit from {self._base_cls} to be a {self._name}"
                    )
            self._objs[alias] = target
            if metadata:
                self._meta[alias] = metadata

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

    # ------------------------------------------------------------------
    # Lookup & materialisation
    # ------------------------------------------------------------------

    @lru_cache(maxsize=256)
    def _materialise(self, ph: Placeholder) -> T:
        if isinstance(ph, str):
            mod, _, attr = ph.partition(":")
            if not attr:
                raise ValueError(f"Invalid lazy path '{ph}', expected 'module:object'")
            return cast(T, getattr(importlib.import_module(mod), attr))
        return cast(T, ph.load())

    def get(self, alias: str) -> T:
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

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def __getitem__(self, alias: str) -> T:  # noqa: DunderImplemented
        return self.get(alias)

    def __iter__(self):  # noqa: DunderImplemented
        return iter(self._objs)

    def __len__(self):  # noqa: DunderImplemented
        return len(self._objs)

    def items(self):  # noqa: DunderImplemented
        return self._objs.items()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def metadata(self, alias: str) -> Union[Mapping[str, Any], None]:
        return self._meta.get(alias)

    def origin(self, alias: str) -> Union[str, None]:
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
        with self._lock:
            self._objs = MappingProxyType(dict(self._objs))  # type: ignore[assignment]
            self._meta = MappingProxyType(dict(self._meta))  # type: ignore[assignment]

    # Test helper -------------------------------------------------------------

    def _clear(self):  # pragma: no cover
        """Erase registry (for isolated tests)."""
        self._objs.clear()
        self._meta.clear()
        self._materialise.cache_clear()


# ────────────────────────────────────────────────────────────────────────
# Structured object for metrics
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    compute: Callable[[Any, Any], Any]
    aggregate: Callable[[Iterable[Any]], float]
    higher_is_better: bool = True
    output_type: Union[str, None] = None
    requires: Union[list[str], None] = None


# ────────────────────────────────────────────────────────────────────────
# Canonical registries
# ────────────────────────────────────────────────────────────────────────

from lm_eval.api.model import LM  # noqa: E402


model_registry: Registry[type[LM]] = Registry("model", base_cls=LM)
task_registry: Registry[Callable[..., Any]] = Registry("task")
metric_registry: Registry[MetricSpec] = Registry("metric")
metric_agg_registry: Registry[Callable[[Iterable[Any]], float]] = Registry(
    "metric aggregation"
)
higher_is_better_registry: Registry[bool] = Registry("higher‑is‑better flag")
filter_registry: Registry[Callable] = Registry("filter")

# Public helper aliases ------------------------------------------------------

register_model = model_registry.register
get_model = model_registry.get

register_task = task_registry.register
get_task = task_registry.get

register_filter = filter_registry.register
get_filter = filter_registry.get

# Metric helpers need thin wrappers to build MetricSpec ----------------------


def register_metric(**kw):
    name = kw["metric"]

    def deco(fn):
        spec = MetricSpec(
            compute=fn,
            aggregate=(
                metric_agg_registry.get(kw["aggregation"])
                if "aggregation" in kw
                else lambda _: {}
            ),
            higher_is_better=kw.get("higher_is_better", True),
            output_type=kw.get("output_type"),
            requires=kw.get("requires"),
        )
        metric_registry.register(name, lazy=spec, metadata=kw)
        higher_is_better_registry.register(name, lazy=spec.higher_is_better)
        return fn

    return deco


def get_metric(name, hf_evaluate_metric=False):
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
            raise KeyError(f"Metric '{name}' not found anywhere")


register_metric_aggregation = metric_agg_registry.register
get_metric_aggregation = metric_agg_registry.get

register_higher_is_better = higher_is_better_registry.register
is_higher_better = higher_is_better_registry.get

# Legacy compatibility
register_aggregation = metric_agg_registry.register
get_aggregation = metric_agg_registry.get
DEFAULT_METRIC_REGISTRY = metric_registry
AGGREGATION_REGISTRY = metric_agg_registry

# Convenience ----------------------------------------------------------------


def freeze_all():
    for r in (
        model_registry,
        task_registry,
        metric_registry,
        metric_agg_registry,
        higher_is_better_registry,
        filter_registry,
    ):
        r.freeze()


# Backwards‑compat read‑only aliases ----------------------------------------

MODEL_REGISTRY = model_registry  # type: ignore
TASK_REGISTRY = task_registry  # type: ignore
METRIC_REGISTRY = metric_registry  # type: ignore
METRIC_AGGREGATION_REGISTRY = metric_agg_registry  # type: ignore
HIGHER_IS_BETTER_REGISTRY = higher_is_better_registry  # type: ignore
FILTER_REGISTRY = filter_registry  # type: ignore
