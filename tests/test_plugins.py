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


def test_import_plugins_tolerates_bad_module():
    # Should log and continue, not raise.
    import_plugins(["definitely_not_a_real_module_xyz"])


def test_import_plugins_noop_on_empty():
    import_plugins(None)
    import_plugins([])
