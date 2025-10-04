#!/usr/bin/env python3
"""Comprehensive tests for the registry system."""

import threading

import pytest

from lm_eval.api.model import LM
from lm_eval.api.registry import (
    MetricSpec,
    Registry,
    get_metric,
    metric_agg_registry,
    metric_registry,
    model_registry,
    register_metric,
)


# Import metrics module to ensure decorators are executed
# import lm_eval.api.metrics


class TestBasicRegistry:
    """Test basic registry functionality."""

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
        assert reg.get("my_class") == MyClass
        assert reg["my_class"] == MyClass

    def test_decorator_multiple_aliases(self):
        """Test decorator with multiple aliases."""
        reg = Registry("test")

        @reg.register("alias1", "alias2", "alias3")
        class MyClass:
            pass

        assert reg.get("alias1") == MyClass
        assert reg.get("alias2") == MyClass
        assert reg.get("alias3") == MyClass

    def test_decorator_auto_name(self):
        """Test decorator using class name when no alias provided."""
        reg = Registry("test")

        @reg.register()
        class AutoNamedClass:
            pass

        assert reg.get("AutoNamedClass") == AutoNamedClass

    def test_lazy_registration(self):
        """Test lazy loading with module paths."""
        reg = Registry("test")

        # Register with lazy loading
        reg.register("join", lazy="os.path:join")

        # Check it's stored as a string
        assert isinstance(reg._objs["join"], str)

        # Access triggers materialization
        result = reg.get("join")
        import os

        assert result == os.path.join
        assert callable(result)

    def test_direct_registration(self):
        """Test direct object registration."""
        reg = Registry("test")

        class DirectClass:
            pass

        obj = DirectClass()
        reg.register("direct", lazy=obj)

        assert reg.get("direct") == obj

    def test_metadata_removed(self):
        """Test that metadata parameter is removed from generic registry."""
        reg = Registry("test")

        # Should work without metadata parameter
        @reg.register("test_class")
        class TestClass:
            pass

        assert "test_class" in reg
        assert reg.get("test_class") == TestClass

    def test_unknown_key_error(self):
        """Test error when accessing unknown key."""
        reg = Registry("test")

        with pytest.raises(KeyError) as exc_info:
            reg.get("unknown")

        assert "Unknown test 'unknown'" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_iteration(self):
        """Test registry iteration."""
        reg = Registry("test")

        reg.register("a", lazy="os:getcwd")
        reg.register("b", lazy="os:getenv")
        reg.register("c", lazy="os:getpid")

        assert list(reg) == ["a", "b", "c"]
        assert len(reg) == 3

        # Test items()
        items = list(reg.items())
        assert len(items) == 3
        assert items[0][0] == "a"
        assert isinstance(items[0][1], str)  # Still lazy

    def test_mapping_protocol(self):
        """Test that registry implements mapping protocol."""
        reg = Registry("test")

        reg.register("test", lazy="os:getcwd")

        # __getitem__
        assert reg["test"] == reg.get("test")

        # __contains__
        assert "test" in reg
        assert "missing" not in reg

        # __iter__ and __len__ tested above


class TestTypeConstraints:
    """Test type checking and base class constraints."""

    def test_base_class_constraint(self):
        """Test base class validation."""

        # Define a base class
        class BaseClass:
            pass

        class GoodSubclass(BaseClass):
            pass

        class BadClass:
            pass

        reg = Registry("typed", base_cls=BaseClass)

        # Should work - correct subclass
        @reg.register("good")
        class GoodInline(BaseClass):
            pass

        # Should fail - wrong type
        with pytest.raises(TypeError) as exc_info:

            @reg.register("bad")
            class BadInline:
                pass

        assert "must inherit from" in str(exc_info.value)

    def test_lazy_type_check(self):
        """Test that type checking happens on materialization for lazy entries."""

        class BaseClass:
            pass

        reg = Registry("typed", base_cls=BaseClass)

        # Register a lazy entry that will fail type check
        reg.register("bad_lazy", lazy="os.path:join")

        # Should fail when accessed - the error message varies
        with pytest.raises(TypeError):
            reg.get("bad_lazy")


class TestCollisionHandling:
    """Test registration collision scenarios."""

    def test_identical_registration(self):
        """Test that identical re-registration is allowed."""
        reg = Registry("test")

        class MyClass:
            pass

        # First registration
        reg.register("test", lazy=MyClass)

        # Identical re-registration should work
        reg.register("test", lazy=MyClass)

        assert reg.get("test") == MyClass

    def test_different_registration_fails(self):
        """Test that different re-registration fails."""
        reg = Registry("test")

        class Class1:
            pass

        class Class2:
            pass

        reg.register("test", lazy=Class1)

        with pytest.raises(ValueError) as exc_info:
            reg.register("test", lazy=Class2)

        assert "already registered" in str(exc_info.value)

    def test_lazy_to_concrete_upgrade(self):
        """Test that lazy placeholder can be upgraded to concrete class."""
        reg = Registry("test")

        # Register lazy
        reg.register("myclass", lazy="test_registry:MyUpgradeClass")

        # Define and register concrete - should work
        @reg.register("myclass")
        class MyUpgradeClass:
            pass

        assert reg.get("myclass") == MyUpgradeClass


class TestThreadSafety:
    """Test thread safety of registry operations."""

    def test_concurrent_access(self):
        """Test concurrent access to lazy entries."""
        reg = Registry("test")

        # Register lazy entry
        reg.register("concurrent", lazy="os.path:join")

        results = []
        errors = []

        def access_item():
            try:
                result = reg.get("concurrent")
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Launch threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=access_item)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 10
        # All should get the same object
        assert all(r == results[0] for r in results)

    def test_concurrent_registration(self):
        """Test concurrent registration doesn't cause issues."""
        reg = Registry("test")

        errors = []

        def register_item(name, value):
            try:
                reg.register(name, lazy=value)
            except Exception as e:
                errors.append(str(e))

        # Launch threads with different registrations
        threads = []
        for i in range(10):
            t = threading.Thread(
                target=register_item, args=(f"item_{i}", f"module{i}:Class{i}")
            )
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0
        assert len(reg) == 10


class TestMetricRegistry:
    """Test metric-specific registry functionality."""

    def test_metric_spec(self):
        """Test MetricSpec dataclass."""

        def compute_fn(items):
            return [1 for _ in items]

        def agg_fn(values):
            return sum(values) / len(values)

        spec = MetricSpec(
            compute=compute_fn,
            aggregate=agg_fn,
            higher_is_better=True,
            output_type="probability",
        )

        assert spec.compute == compute_fn
        assert spec.aggregate == agg_fn
        assert spec.higher_is_better
        assert spec.output_type == "probability"

    def test_register_metric_decorator(self):
        """Test @register_metric decorator."""

        # Register aggregation function first
        @metric_agg_registry.register("test_mean")
        def test_mean(values):
            return sum(values) / len(values) if values else 0.0

        # Register metric
        @register_metric(
            metric="test_accuracy",
            aggregation="test_mean",
            higher_is_better=True,
            output_type="accuracy",
        )
        def compute_accuracy(items):
            return [1 if item["pred"] == item["gold"] else 0 for item in items]

        # Check registration
        assert "test_accuracy" in metric_registry
        spec = metric_registry.get("test_accuracy")
        assert isinstance(spec, MetricSpec)
        assert spec.higher_is_better
        assert spec.output_type == "accuracy"

        # Test compute function
        items = [
            {"pred": "a", "gold": "a"},
            {"pred": "b", "gold": "b"},
            {"pred": "c", "gold": "d"},
        ]
        result = spec.compute(items)
        assert result == [1, 1, 0]

        # Test aggregation
        agg_result = spec.aggregate(result)
        assert agg_result == 2 / 3

    def test_metric_without_aggregation(self):
        """Test metric registration without aggregation."""

        @register_metric(metric="no_agg", higher_is_better=False)
        def compute_something(items):
            return [len(item) for item in items]

        spec = metric_registry.get("no_agg")

        # Should raise NotImplementedError when aggregate is called
        with pytest.raises(NotImplementedError) as exc_info:
            spec.aggregate([1, 2, 3])

        assert "No aggregation function specified" in str(exc_info.value)

    def test_get_metric_helper(self):
        """Test get_metric helper function."""

        @register_metric(
            metric="helper_test",
            aggregation="mean",  # Assuming 'mean' exists in metric_agg_registry
        )
        def compute_helper(items):
            return items

        # get_metric returns just the compute function
        compute_fn = get_metric("helper_test")
        assert callable(compute_fn)
        assert compute_fn([1, 2, 3]) == [1, 2, 3]


class TestRegistryUtilities:
    """Test utility methods."""

    def test_freeze(self):
        """Test freezing a registry."""
        reg = Registry("test")

        # Add some items
        reg.register("item1", lazy="os:getcwd")
        reg.register("item2", lazy="os:getenv")

        # Freeze the registry
        reg.freeze()

        # Should not be able to register new items
        with pytest.raises(TypeError):
            reg._objs["new"] = "value"

        # Should still be able to access items
        assert "item1" in reg
        assert callable(reg.get("item1"))

    def test_clear(self):
        """Test clearing a registry."""
        reg = Registry("test")

        # Add items
        reg.register("item1", lazy="os:getcwd")
        reg.register("item2", lazy="os:getenv")

        assert len(reg) == 2

        # Clear
        reg._clear()

        assert len(reg) == 0
        assert list(reg) == []

    def test_origin(self):
        """Test origin tracking."""
        reg = Registry("test")

        # Lazy entry - no origin
        reg.register("lazy", lazy="os:getcwd")
        assert reg.origin("lazy") is None

        # Concrete class - should have origin
        @reg.register("concrete")
        class ConcreteClass:
            pass

        origin = reg.origin("concrete")
        assert origin is not None
        assert "test_registry.py" in origin
        assert ":" in origin  # Has line number


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_model_registry_alias(self):
        """Test MODEL_REGISTRY backward compatibility."""
        from lm_eval.api.registry import MODEL_REGISTRY

        # Should be same object as model_registry
        assert MODEL_REGISTRY is model_registry

        # Should reflect current state
        before_count = len(MODEL_REGISTRY)

        # Add new model
        @model_registry.register("test_model_compat")
        class TestModelCompat(LM):
            pass

        # MODEL_REGISTRY should immediately reflect the change
        assert len(MODEL_REGISTRY) == before_count + 1
        assert "test_model_compat" in MODEL_REGISTRY

    def test_legacy_functions(self):
        """Test legacy helper functions."""
        from lm_eval.api.registry import (
            AGGREGATION_REGISTRY,
            DEFAULT_METRIC_REGISTRY,
            get_model,
            register_model,
        )

        # register_model should work
        @register_model("legacy_model")
        class LegacyModel(LM):
            pass

        # get_model should work
        assert get_model("legacy_model") == LegacyModel

        # Check other aliases
        assert DEFAULT_METRIC_REGISTRY is metric_registry
        assert AGGREGATION_REGISTRY is metric_agg_registry


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_lazy_format(self):
        """Test error on invalid lazy format."""
        reg = Registry("test")

        reg.register("bad", lazy="no_colon_here")

        with pytest.raises(ValueError) as exc_info:
            reg.get("bad")

        assert "expected 'module:object'" in str(exc_info.value)

    def test_lazy_module_not_found(self):
        """Test error when lazy module doesn't exist."""
        reg = Registry("test")

        reg.register("missing", lazy="nonexistent_module:Class")

        with pytest.raises(ModuleNotFoundError):
            reg.get("missing")

    def test_lazy_attribute_not_found(self):
        """Test error when lazy attribute doesn't exist."""
        reg = Registry("test")

        reg.register("missing_attr", lazy="os:nonexistent_function")

        with pytest.raises(AttributeError):
            reg.get("missing_attr")

    def test_multiple_aliases_with_lazy(self):
        """Test that multiple aliases with lazy fails."""
        reg = Registry("test")

        with pytest.raises(ValueError) as exc_info:
            reg.register("alias1", "alias2", lazy="os:getcwd")

        assert "Exactly one alias required" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
