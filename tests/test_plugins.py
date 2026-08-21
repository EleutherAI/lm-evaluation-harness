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


@pytest.fixture(autouse=True)
def _reset_plugin_state(monkeypatch):
    """Ensure the once-per-group guard is clean for every test."""
    monkeypatch.setattr(registry_mod, "_loaded_plugin_groups", set())
    yield


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
    assert reg.get("good") is good


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
    """A plugin that declares BOTH an entry point and @register_model must not
    collide: materializing the EntryPoint imports the module whose decorator
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
    from lm_eval.api.registry import filter_registry, get_filter

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

    try:
        assert get_filter("plugin-filter") is PluginFilter
    finally:
        filter_registry._objs.pop("plugin-filter", None)


def test_get_aggregation_discovers_entry_point(monkeypatch):
    """get_aggregation resolves a function from the lm_eval.aggregations group."""
    from lm_eval.api.registry import aggregation_registry, get_aggregation

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

    try:
        # Metrics module must be importable for the len()==0 guard; it is.
        assert get_aggregation("plugin-agg") is plugin_agg
    finally:
        aggregation_registry._objs.pop("plugin-agg", None)


def test_metric_plugin_populates_side_effect_registries(monkeypatch):
    """A metric plugin declared as an entry point, whose module uses
    @register_metric, must populate higher_is_better and aggregation registries
    once materialized via is_higher_better / get_metric_aggregation.
    """
    from lm_eval.api.registry import (
        get_metric,
        get_metric_aggregation,
        is_higher_better,
        metric_agg_registry,
        metric_registry,
    )

    # Build a module whose import registers a metric with side-effect metadata.
    mod = types.ModuleType(_FAKE_MODULE + "_metric")

    def _register():
        from lm_eval.api.registry import register_aggregation, register_metric

        @register_aggregation("plugin-metric-agg")
        def _agg(items):
            return sum(items) / len(items)

        @register_metric(
            metric="plugin-metric",
            higher_is_better=True,
            aggregation="plugin-metric-agg",
        )
        def _metric(items):
            return sum(items)

        mod.plugin_metric = _metric

    _register()  # run the decorators now; entry point just points at the function
    ep = registry_mod.md.EntryPoint(
        name="plugin-metric",
        value=f"{mod.__name__}:plugin_metric",
        group="lm_eval.metrics",
    )
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    _patch_entry_points(monkeypatch, "lm_eval.metrics", [ep])

    try:
        assert callable(get_metric("plugin-metric"))
        # These read the side-effect registries populated by @register_metric.
        assert is_higher_better("plugin-metric") is True
        assert get_metric_aggregation("plugin-metric") is not None
    finally:
        metric_registry._objs.pop("plugin-metric", None)
        metric_agg_registry._objs.pop("plugin-metric", None)


def test_import_plugins_tolerates_bad_module():
    # Should log and continue, not raise.
    import_plugins(["definitely_not_a_real_module_xyz"])


def test_import_plugins_noop_on_empty():
    import_plugins(None)
    import_plugins([])
