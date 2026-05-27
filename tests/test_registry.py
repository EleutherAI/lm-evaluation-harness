"""Tests for the registry system."""

import threading

import pytest

from lm_eval.api.registry import (
    Registry,
    aggregation_registry,
    filter_registry,
    get_aggregation,
    get_filter,
    get_metric,
    get_metric_aggregation,
    get_model,
    higher_is_better_registry,
    is_higher_better,
    metric_agg_registry,
    metric_registry,
    model_registry,
    register_aggregation,
    register_filter,
    register_metric,
)


class TestRegistryBasics:
    """Test basic Registry class functionality."""

    def test_create_registry(self):
        """Test creating a basic registry."""
        reg = Registry("test")
        assert len(reg) == 0
        assert list(reg) == []

    def test_decorator_registration(self):
        """Test decorator-based registration."""
        reg = Registry("test")

        @reg.register("my_class")
        class MyClass:
            pass

        assert "my_class" in reg
        assert reg.get("my_class") is MyClass
        assert reg["my_class"] is MyClass

    def test_decorator_multiple_aliases(self):
        """Test decorator with multiple aliases."""
        reg = Registry("test")

        @reg.register("alias1", "alias2", "alias3")
        class MyClass:
            pass

        assert reg.get("alias1") is MyClass
        assert reg.get("alias2") is MyClass
        assert reg.get("alias3") is MyClass

    def test_decorator_auto_name(self):
        """Test decorator using class name when no alias provided."""
        reg = Registry("test")

        @reg.register()
        class AutoNamedClass:
            pass

        assert reg.get("AutoNamedClass") is AutoNamedClass

    def test_lazy_registration(self):
        """Test lazy loading with module paths."""
        reg = Registry("test")

        # Register with lazy loading
        reg.register("join", target="os.path:join")

        # Check it's stored as a string (placeholder)
        assert isinstance(reg._objs["join"], str)

        # Access triggers materialization
        import os.path

        result = reg.get("join")
        assert result is os.path.join
        assert callable(result)

    def test_unknown_key_error(self):
        """Test error when accessing unknown key."""
        reg = Registry("test")

        with pytest.raises(KeyError) as exc_info:
            reg.get("unknown")

        assert "Unknown test 'unknown'" in str(exc_info.value)

    def test_default_value(self):
        """Test default value when key not found."""
        reg = Registry("test")

        assert reg.get("missing", "default") == "default"
        assert reg.get("missing", None) is None
        assert reg.get("missing", 0) == 0

    def test_iteration(self):
        """Test registry iteration."""
        reg = Registry("test")

        reg.register("a", target="os:getcwd")
        reg.register("b", target="os:getenv")
        reg.register("c", target="os:getpid")

        assert set(reg) == {"a", "b", "c"}
        assert len(reg) == 3

    def test_contains(self):
        """Test 'in' operator."""
        reg = Registry("test")
        reg.register("exists", target="os:getcwd")

        assert "exists" in reg
        assert "missing" not in reg

    def test_keys_values_items(self):
        """Test dict-like methods."""
        reg = Registry("test")
        reg.register("a", target="os:getcwd")
        reg.register("b", target="os:getenv")

        assert set(reg.keys()) == {"a", "b"}
        assert len(list(reg.values())) == 2
        assert len(list(reg.items())) == 2


class TestRegistryCollisions:
    """Test collision handling in Registry."""

    def test_duplicate_raises_error(self):
        """Test that registering different objects under same alias raises error."""
        reg = Registry("test")

        @reg.register("name")
        class First:
            pass

        with pytest.raises(ValueError) as exc_info:

            @reg.register("name")
            class Second:
                pass

        assert "already registered" in str(exc_info.value)

    def test_placeholder_upgrade(self):
        """Test that placeholder can be upgraded to concrete class."""
        reg = Registry("test")

        # Create a class to test with
        class MyTestClass:
            pass

        # Register as placeholder with correct module:class path
        placeholder_path = f"{MyTestClass.__module__}:{MyTestClass.__name__}"
        reg.register("my_alias", target=placeholder_path)

        # Registering the actual class should upgrade the placeholder
        reg.register("my_alias")(MyTestClass)

        assert reg.get("my_alias") is MyTestClass

    def test_same_object_no_error(self):
        """Test that registering same object twice doesn't raise error."""
        reg = Registry("test")

        class MyClass:
            pass

        reg.register("name")(MyClass)
        reg.register("name")(MyClass)  # Should not raise

        assert reg.get("name") is MyClass


class TestRegistryFreeze:
    """Test registry freezing."""

    def test_freeze(self):
        """Test that freeze makes registry read-only."""
        reg = Registry("test")
        reg.register("a", target="os:getcwd")
        reg.freeze()

        # Can still read
        assert "a" in reg

        # But modifications fail
        with pytest.raises(TypeError):
            reg.register("b", target="os:getenv")

    def test_freeze_all(self):
        """Test freeze_all function."""

        # This would freeze all global registries - skip in test
        # freeze_all()
        pass


class TestRegistryThreadSafety:
    """Test thread safety of Registry."""

    def test_concurrent_registration(self):
        """Test concurrent registration from multiple threads."""
        reg = Registry("test")
        errors = []

        def register_item(i):
            try:
                reg.register(f"item_{i}", target="os:getcwd")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_item, args=(i,)) for i in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(reg) == 100

    def test_concurrent_access(self):
        """Test concurrent access from multiple threads."""
        reg = Registry("test")
        reg.register("item", target="os.path:join")

        results = []

        def access_item():
            result = reg.get("item")
            results.append(result)

        threads = [threading.Thread(target=access_item) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same result
        import os.path

        assert all(r is os.path.join for r in results)


class TestModelRegistry:
    """Test model registry integration."""

    def test_model_registry_exists(self):
        """Test that model_registry is properly initialized."""
        assert model_registry is not None

    def test_lazy_model_loading(self):
        """Test lazy loading of models.

        get_model() auto-imports lm_eval.models if registry is empty,
        so this works without explicit import.
        """
        # get_model auto-imports models if registry is empty
        dummy_cls = get_model("dummy")
        assert dummy_cls is not None
        assert "DummyLM" in dummy_cls.__name__

        # After get_model, registry should be populated
        assert "dummy" in model_registry
        assert "hf" in model_registry

    def test_get_model_error(self):
        """Test get_model with unknown model."""
        with pytest.raises(ValueError) as exc_info:
            get_model("nonexistent_model_xyz")

        assert "no model for this name found" in str(exc_info.value)


class TestFilterRegistry:
    """Test filter registry integration."""

    def test_filter_registry_exists(self):
        """Test that filter_registry is properly initialized."""
        assert filter_registry is not None

    def test_register_filter(self):
        """Test registering a filter."""
        from lm_eval.api.filter import Filter

        @register_filter("test_filter_unique")
        class TestFilter(Filter):
            def apply(self, resps, docs):
                return resps

        assert "test_filter_unique" in filter_registry
        assert get_filter("test_filter_unique") is TestFilter

    def test_get_filter_callable(self):
        """Test get_filter with callable input."""

        def my_filter(x):
            return x

        assert get_filter(my_filter) is my_filter


class TestMetricRegistry:
    """Test metric registry integration."""

    def test_metric_registry_exists(self):
        """Test that metric_registry is properly initialized."""
        assert metric_registry is not None

    def test_aggregation_registry_exists(self):
        """Test that aggregation_registry is properly initialized."""
        assert aggregation_registry is not None

    def test_register_aggregation(self):
        """Test registering an aggregation function."""

        @register_aggregation("test_agg_unique")
        def test_agg(items):
            return sum(items) / len(items)

        assert "test_agg_unique" in aggregation_registry
        assert get_aggregation("test_agg_unique") is test_agg

    def test_register_metric(self):
        """Test registering a metric."""

        # First register the aggregation
        @register_aggregation("test_metric_agg")
        def mean_agg(items):
            return sum(items) / len(items)

        @register_metric(
            metric="test_metric_unique",
            higher_is_better=True,
            aggregation="test_metric_agg",
        )
        def test_metric(items):
            return sum(1 for i in items if i)

        assert "test_metric_unique" in metric_registry
        assert get_metric("test_metric_unique") is test_metric
        assert is_higher_better("test_metric_unique") is True
        assert get_metric_aggregation("test_metric_unique") is mean_agg

    def test_builtin_metrics_loaded(self):
        """Test that built-in metrics are loaded."""
        # Import metrics module to trigger registration
        from lm_eval.api import metrics  # noqa: F401

        # Check some common metrics are registered
        assert "acc" in metric_registry
        assert "mean" in aggregation_registry


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_registry_aliases(self):
        """Test that UPPER_CASE aliases point to Registry instances."""
        from lm_eval.api.registry import (
            AGGREGATION_REGISTRY,
            FILTER_REGISTRY,
            HIGHER_IS_BETTER_REGISTRY,
            METRIC_AGGREGATION_REGISTRY,
            METRIC_REGISTRY,
            MODEL_REGISTRY,
        )

        assert MODEL_REGISTRY is model_registry
        assert FILTER_REGISTRY is filter_registry
        assert METRIC_REGISTRY is metric_registry
        assert AGGREGATION_REGISTRY is aggregation_registry
        assert METRIC_AGGREGATION_REGISTRY is metric_agg_registry
        assert HIGHER_IS_BETTER_REGISTRY is higher_is_better_registry


class TestRegistryClear:
    """Test registry clear functionality (for test isolation)."""

    def test_clear(self):
        """Test _clear method for test isolation."""
        reg = Registry("test")
        reg.register("a", target="os:getcwd")
        reg.register("b", target="os:getenv")

        assert len(reg) == 2

        reg._clear()

        assert len(reg) == 0
        assert "a" not in reg
