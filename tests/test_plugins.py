"""Tests for runtime plugin registration (entry points + --plugins imports)."""

import sys
import types

import pytest

import lm_eval.api.registry as registry_mod
from lm_eval.api.registry import (
    Registry,
    import_plugins,
    load_plugins,
)


_FAKE_MODULE = "lm_eval_fake_plugin_module"


def _make_ep(name, obj, *, broken=False):
    """Build a real EntryPoint whose load() resolves against an in-memory module.

    Using a genuine md.EntryPoint (not a stub) ensures Registry.get() takes its
    materialization path. The value points at an attribute on a module we inject
    into sys.modules; a "broken" entry point points at a missing attribute.
    """
    mod = sys.modules.setdefault(_FAKE_MODULE, types.ModuleType(_FAKE_MODULE))
    attr = f"_ep_{name}".replace("-", "_")
    if not broken:
        setattr(mod, attr, obj)
    return registry_mod.md.EntryPoint(
        name=name, value=f"{_FAKE_MODULE}:{attr}", group="test"
    )


_REGISTRIES = (
    "model_registry",
    "filter_registry",
    "aggregation_registry",
    "metric_registry",
    "metric_agg_registry",
    "higher_is_better_registry",
)


@pytest.fixture(autouse=True)
def _reset_plugin_state(monkeypatch):
    """Undo the process-global state a plugin test mutates.

    Registration is not scoped to a call: discovering a plugin writes into module
    level registries, imports its module into ``sys.modules`` and marks its entry
    point group as scanned. Snapshot and restore all three so tests cannot leak
    into each other regardless of ordering.
    """
    monkeypatch.setattr(registry_mod, "_loaded_plugin_groups", set())
    snapshots = {name: dict(getattr(registry_mod, name)._objs) for name in _REGISTRIES}
    yield
    for name, snapshot in snapshots.items():
        objs = getattr(registry_mod, name)._objs
        objs.clear()
        objs.update(snapshot)
    sys.modules.pop(_FAKE_MODULE, None)
    registry_mod._materialise_placeholder.cache_clear()


@pytest.fixture
def lazy_plugin_module(tmp_path, monkeypatch):
    """Write a plugin module to disk that starts out *unimported*.

    A module pre-seeded into ``sys.modules`` lets a test pass without the entry
    point ever being loaded, which hides exactly the lazy-discovery behaviour
    these tests exist to pin down. Writing a real file and asserting it is absent
    from ``sys.modules`` forces materialization to do the import.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    written: list[str] = []

    def _write(name: str, source: str) -> str:
        (tmp_path / f"{name}.py").write_text(source)
        assert name not in sys.modules, f"{name} must not be imported yet"
        written.append(name)
        return name

    yield _write

    for name in written:
        sys.modules.pop(name, None)


def _patch_entry_points(monkeypatch, group_name, eps):
    def fake_entry_points(*, group):
        return eps if group == group_name else []

    monkeypatch.setattr(registry_mod.md, "entry_points", fake_entry_points)


def test_load_plugins_registers_and_resolves(monkeypatch):
    reg = Registry("thing")
    sentinel = object()
    _patch_entry_points(monkeypatch, "lm_eval.things", [_make_ep("fake", sentinel)])

    discovered = load_plugins("lm_eval.things", reg)

    assert discovered == ["fake"]
    assert "fake" in reg
    # Lazy: resolves through the EntryPoint.load() path on access.
    assert reg.get("fake") is sentinel


def test_load_plugins_is_once_per_group(monkeypatch):
    reg = Registry("thing")
    _patch_entry_points(monkeypatch, "lm_eval.things", [_make_ep("fake", object())])

    assert load_plugins("lm_eval.things", reg) == ["fake"]
    # Second call is a no-op because the group is already marked loaded.
    assert load_plugins("lm_eval.things", reg) == []


def test_load_plugins_does_not_override_builtin(monkeypatch):
    reg = Registry("thing")
    builtin = object()
    reg.register("taken", target="os:getcwd")  # pre-existing alias
    _patch_entry_points(monkeypatch, "lm_eval.things", [_make_ep("taken", builtin)])

    discovered = load_plugins("lm_eval.things", reg)

    assert discovered == []  # collision skipped, not registered
    assert reg._objs["taken"] == "os:getcwd"


def test_load_plugins_broken_entry_point_is_tolerated(monkeypatch):
    reg = Registry("thing")
    good = object()
    _patch_entry_points(
        monkeypatch,
        "lm_eval.things",
        [
            _make_ep("bad", None, broken=True),
            _make_ep("good", good),
        ],
    )

    # A broken plugin must not prevent registration; registration itself won't
    # raise (EntryPoint is stored lazily), so both names register but resolving
    # the broken one raises on access while the good one still works.
    discovered = load_plugins("lm_eval.things", reg)

    assert "good" in discovered
    assert "bad" in discovered
    assert reg.get("good") is good
    # The broken plugin surfaces its failure only on access, not at discovery.
    with pytest.raises(AttributeError):
        reg.get("bad")


def test_import_plugins_runs_register_decorator(monkeypatch):
    """import_plugins imports a module, triggering its @register_model."""
    from lm_eval.api.model import LM
    from lm_eval.api.registry import model_registry, register_model

    # Build a module whose import side effect registers a model, then map it into
    # sys.modules so import_plugins resolves it without touching disk.
    mod = types.ModuleType("fake_plugin_pkg")

    def _register():
        @register_model("plugin-test-model")
        class PluginTestLM(LM):
            def loglikelihood(self, requests):
                return []

            def loglikelihood_rolling(self, requests):
                return []

            def generate_until(self, requests):
                return []

    monkeypatch.setitem(sys.modules, "fake_plugin_pkg", mod)
    # import_module returns the pre-built module; run its "import side effect" here.
    monkeypatch.setattr(
        registry_mod.importlib,
        "import_module",
        lambda name: (_register(), sys.modules[name])[1],
    )

    try:
        import_plugins(["fake_plugin_pkg"])
        assert "plugin-test-model" in model_registry
        assert issubclass(model_registry.get("plugin-test-model"), LM)
    finally:
        model_registry._objs.pop("plugin-test-model", None)


def test_entrypoint_upgrades_to_decorated_class(monkeypatch):
    """A plugin that declares BOTH an entry point and @register_model must not collide.

    Materializing the EntryPoint imports the module whose decorator
    re-registers the same alias, and that should upgrade the placeholder.
    """
    reg = Registry("thing")

    class Plugin:
        pass

    # Register the alias as an EntryPoint placeholder pointing at Plugin's path.
    mod = sys.modules.setdefault(_FAKE_MODULE, types.ModuleType(_FAKE_MODULE))
    mod.Plugin = Plugin
    Plugin.__module__ = _FAKE_MODULE
    ep = registry_mod.md.EntryPoint(
        name="dual", value=f"{_FAKE_MODULE}:Plugin", group="test"
    )
    reg.register("dual", target=ep)

    # Simulate the decorator firing (as it would on import) - must not raise.
    reg.register("dual")(Plugin)

    assert reg.get("dual") is Plugin


def test_get_filter_discovers_entry_point(monkeypatch):
    """get_filter resolves a filter contributed via the lm_eval.filters group."""
    from lm_eval.api.filter import Filter
    from lm_eval.api.registry import get_filter

    class PluginFilter(Filter):
        def apply(self, resps, docs):
            return resps

    mod = sys.modules.setdefault(_FAKE_MODULE, types.ModuleType(_FAKE_MODULE))
    mod.PluginFilter = PluginFilter
    PluginFilter.__module__ = _FAKE_MODULE
    ep = registry_mod.md.EntryPoint(
        name="plugin-filter",
        value=f"{_FAKE_MODULE}:PluginFilter",
        group="lm_eval.filters",
    )
    _patch_entry_points(monkeypatch, "lm_eval.filters", [ep])

    assert get_filter("plugin-filter") is PluginFilter


def test_get_aggregation_discovers_entry_point(monkeypatch):
    """get_aggregation resolves a function from the lm_eval.aggregations group."""
    from lm_eval.api.registry import get_aggregation

    def plugin_agg(items):
        return sum(items)

    mod = sys.modules.setdefault(_FAKE_MODULE, types.ModuleType(_FAKE_MODULE))
    mod.plugin_agg = plugin_agg
    plugin_agg.__module__ = _FAKE_MODULE
    ep = registry_mod.md.EntryPoint(
        name="plugin-agg",
        value=f"{_FAKE_MODULE}:plugin_agg",
        group="lm_eval.aggregations",
    )
    _patch_entry_points(monkeypatch, "lm_eval.aggregations", [ep])

    # Metrics module must be importable for the len()==0 guard; it is.
    assert get_aggregation("plugin-agg") is plugin_agg


_METRIC_PLUGIN_SOURCE = """
from lm_eval.api.registry import register_aggregation, register_metric


@register_aggregation("plugin-metric-agg")
def plugin_agg(items):
    return sum(items) / len(items)


@register_metric(
    metric="plugin-metric",
    higher_is_better=True,
    aggregation="plugin-metric-agg",
)
def plugin_metric(items):
    return sum(items)
"""


def test_metric_plugin_populates_side_effect_registries(
    monkeypatch, lazy_plugin_module
):
    """A metric contributed only as an entry point must still carry its metadata.

    ``@register_metric`` fills the higher_is_better and metric-aggregation
    registries as a decorator side effect, which for a lazy plugin only happens
    once its module is imported. ``is_higher_better`` / ``get_metric_aggregation``
    must therefore force discovery and materialization themselves.
    """
    from lm_eval.api.registry import (
        get_metric,
        get_metric_aggregation,
        is_higher_better,
    )

    modname = lazy_plugin_module("lm_eval_fake_metric_plugin", _METRIC_PLUGIN_SOURCE)
    ep = registry_mod.md.EntryPoint(
        name="plugin-metric",
        value=f"{modname}:plugin_metric",
        group="lm_eval.metrics",
    )
    _patch_entry_points(monkeypatch, "lm_eval.metrics", [ep])

    # Reached without get_metric() having run first: the metadata lookups are the
    # only thing that can trigger the import.
    assert is_higher_better("plugin-metric") is True
    assert get_metric_aggregation("plugin-metric") is not None
    assert callable(get_metric("plugin-metric"))


def test_entrypoint_upgrades_to_decorated_function(monkeypatch, lazy_plugin_module):
    """The placeholder upgrade must accept functions, not just classes.

    Loading the entry point imports the plugin module, whose ``@register_metric``
    immediately re-registers the same alias. For a metric that target is a
    function, so gating the upgrade on ``isinstance(target, type)`` turns the
    normal plugin path into a spurious "already registered" collision.
    """
    from lm_eval.api.registry import get_metric

    modname = lazy_plugin_module("lm_eval_fake_upgrade_plugin", _METRIC_PLUGIN_SOURCE)
    ep = registry_mod.md.EntryPoint(
        name="plugin-metric",
        value=f"{modname}:plugin_metric",
        group="lm_eval.metrics",
    )
    _patch_entry_points(monkeypatch, "lm_eval.metrics", [ep])

    metric = get_metric("plugin-metric")
    assert metric is sys.modules[modname].plugin_metric
    # Placeholder was replaced by the concrete function, not left as an EntryPoint.
    assert not isinstance(
        registry_mod.metric_registry._objs["plugin-metric"], registry_mod.md.EntryPoint
    )


_SHADOWING_PLUGIN_SOURCE = """
from lm_eval.api.registry import register_aggregation, register_metric


@register_aggregation("shadow-agg")
def shadow_agg(items):
    return 0.0


# Deliberately contradicts core's acc (higher_is_better=True) so that reading the
# metadata is enough to tell which registration won.
@register_metric(metric="acc", higher_is_better=False, aggregation="shadow-agg")
def shadow_metric(items):
    return 0.0
"""


def test_builtin_metric_wins_over_plugin_of_same_name(monkeypatch, lazy_plugin_module):
    """Core components are never shadowed by a plugin claiming the same alias."""
    from lm_eval.api.registry import get_metric, is_higher_better

    modname = lazy_plugin_module("lm_eval_fake_shadow_plugin", _SHADOWING_PLUGIN_SOURCE)
    ep = registry_mod.md.EntryPoint(
        name="acc", value=f"{modname}:shadow_metric", group="lm_eval.metrics"
    )
    _patch_entry_points(monkeypatch, "lm_eval.metrics", [ep])

    # The plugin declares higher_is_better=False; core declares True.
    assert is_higher_better("acc") is True
    assert get_metric("acc").__name__ == "acc_fn"
    assert modname not in sys.modules, "a shadowed plugin must never be imported"


def test_import_plugins_tolerates_bad_module():
    # Should log and continue, not raise.
    import_plugins(["definitely_not_a_real_module_xyz"])


def test_import_plugins_noop_on_empty():
    import_plugins(None)
    import_plugins([])
