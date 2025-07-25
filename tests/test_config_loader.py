"""
Tests for the config loader pure functions.

Note: _import_function uses LRU caching, so file changes during runtime
won't be detected unless the cache is cleared.

Test coverage:
- _mk_function_ctor:
  - test_mk_function_ctor_with_resolve_false: no-op lambda when resolve=False
  - test_mk_function_ctor_with_resolve_true: actual function import when resolve=True
- _make_loader:
  - test_make_loader_creates_loader_class: creates YAML loader with !function support
  - test_make_loader_caching: loader classes cached by parameters
- _import_function:
  - test_import_local_module: imports from local .py files
  - test_import_nested_local_module: handles dot-separated nested paths
  - test_import_standard_module: falls back to standard library imports
  - test_import_caching: LRU cache behavior
  - test_import_mtime_sensitivity: cache behavior with file changes
- load():
  - test_load_simple_yaml: basic YAML parsing
  - test_load_with_function_resolved: !function tags resolved to callables
  - test_load_with_function_not_resolved: !function tags become strings
  - test_load_with_includes: include files merged, main values win
  - test_load_with_absolute_include: absolute path includes
  - test_load_without_includes_resolution: includes preserved when disabled
  - test_load_include_cycle_detection: circular includes raise error
  - test_load_multiple_includes: include order precedence (later includes override earlier, main overrides all)
  - test_load_recursive_includes: nested includes (main->inc1->inc2, main overrides inc1 overrides inc2)
  - test_load_expanduser_path: ~ paths expanded
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lm_eval.tasks._config_loader import (
    _Base,
    _import_func_in_yml,
    _make_loader,
    _mk_function_ctor,
    import_fun_from_str,
    load_yaml,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def yaml_file(temp_dir):
    def _create_yaml(content, filename="test.yaml"):
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path

    return _create_yaml


@pytest.fixture
def python_module(temp_dir):
    def _create_module(content, filename="utils.py"):
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path

    return _create_module


class TestMkFunctionCtor:
    """Tests for the YAML !function constructor factory."""

    def test_mk_function_ctor_with_resolve_false(self, temp_dir):
        """When resolve=False, should return a string."""
        ctor = _mk_function_ctor(temp_dir, resolve=False)

        loader = MagicMock()
        node = MagicMock()
        loader.construct_scalar.return_value = "module.function"

        result = ctor(loader, node)

        assert isinstance(result, str)

    def test_mk_function_ctor_with_resolve_true(self, temp_dir, python_module):
        """When resolve=True, should import and return the actual function."""
        # Create a local module
        python_module("def test_func(x):\n    return x * 2\n")

        ctor = _mk_function_ctor(temp_dir, resolve=True)

        loader = MagicMock()
        node = MagicMock()
        loader.construct_scalar.return_value = "utils.test_func"

        result = ctor(loader, node)

        assert callable(result)
        assert result(5) == 10


class TestMakeLoader:
    """Tests for YAML loader class creation and caching."""

    def test_make_loader_creates_loader_class(self, temp_dir):
        loader_cls = _make_loader(temp_dir, resolve_funcs=True)

        assert issubclass(loader_cls, _Base)

        # !function constructor should be registered
        constructors = loader_cls.yaml_constructors
        assert "!function" in constructors

    def test_make_loader_caching(self, temp_dir):
        """Loader classes should be cached by parameters."""
        # Clear cache first
        _make_loader.cache_clear()

        loader1 = _make_loader(temp_dir, resolve_funcs=True)
        loader2 = _make_loader(temp_dir, resolve_funcs=True)
        loader3 = _make_loader(temp_dir, resolve_funcs=False)

        assert loader1 is loader2  # Same params = same class
        assert loader1 is not loader3  # Different params = different class


class TestImportFunction:
    """Tests for dynamic function importing with mtime-based module caching."""

    def test_import_local_module(self, temp_dir, python_module):
        # Create a local module
        python_module("def local_func(x, y):\n    return x + y\n")

        func = _import_func_in_yml("utils.local_func", temp_dir)

        assert callable(func)
        assert func(2, 3) == 5

    def test_import_nested_local_module(self, temp_dir):
        """Should handle dot-separated paths for nested modules."""
        # Create nested directory structure
        (temp_dir / "sub").mkdir()
        (temp_dir / "sub" / "module.py").write_text(
            "def nested_func():\n    return 'nested'\n"
        )

        func = _import_func_in_yml("sub.module.nested_func", temp_dir)

        assert callable(func)
        assert func() == "nested"

    def test_import_standard_module(self, temp_dir):
        """Falls back to standard import for non-local modules."""
        # Import from standard library
        func = _import_func_in_yml("os.path.join", temp_dir)

        assert callable(func)
        assert func("a", "b") in ("a/b", "a\\b")  # Unix or Windows

    def test_import_caching(self, temp_dir, python_module):
        # Clear cache first
        _import_func_in_yml.cache_clear()

        python_module("def cached_func():\n    return 42\n")

        func1 = _import_func_in_yml("utils.cached_func", temp_dir)
        func2 = _import_func_in_yml("utils.cached_func", temp_dir)

        assert func1 is func2  # Cached

    def test_import_mtime_sensitivity(self, temp_dir):
        """Verifies LRU cache behavior - file changes require cache clear."""

        # Clear the LRU cache
        _import_func_in_yml.cache_clear()

        # Create a module
        module_path = temp_dir / "test_mtime.py"
        module_path.write_text("value = 1\n")

        # Import it
        import_key = "test_mtime.value"
        value1 = _import_func_in_yml(import_key, temp_dir)
        assert value1 == 1

        value2 = _import_func_in_yml(import_key, temp_dir)
        assert value2 == 1  # From cache

        _import_func_in_yml.cache_clear()
        value3 = _import_func_in_yml(import_key, temp_dir)
        assert value3 == 1  # Re-imported


class TestImportFunFromStr:
    """Tests for import_fun_from_str function."""

    def test_import_from_absolute_path(self, temp_dir):
        """Test importing function from absolute path."""
        # Create a test module
        module_path = temp_dir / "test_module.py"
        module_path.write_text("def test_func(x):\n    return x * 2\n")

        # Import using absolute path
        func = import_fun_from_str(f"{module_path.with_suffix('')}.test_func")

        assert callable(func)
        assert func(5) == 10

    def test_import_with_py_extension(self, temp_dir):
        """Test importing when .py is included in the path."""
        # Create a test module
        module_path = temp_dir / "test_module.py"
        module_path.write_text("def test_func(x):\n    return x + 10\n")

        # Import with .py in the path
        func = import_fun_from_str(f"{module_path}.test_func")

        assert callable(func)
        assert func(5) == 15

    def test_import_nested_function(self, temp_dir):
        """Test importing from nested module structure."""
        # Create nested directory
        (temp_dir / "subdir").mkdir()
        module_path = temp_dir / "subdir" / "nested.py"
        module_path.write_text("def nested_func():\n    return 'nested'\n")

        # Import from nested path
        func = import_fun_from_str(f"{module_path.with_suffix('')}.nested_func")

        assert callable(func)
        assert func() == "nested"

    def test_import_missing_module(self, temp_dir):
        """Test error when module doesn't exist."""
        with pytest.raises(ImportError, match="Module file not found"):
            import_fun_from_str(f"{temp_dir}/nonexistent.test_func")

    def test_import_missing_function(self, temp_dir):
        """Test error when function doesn't exist in module."""
        module_path = temp_dir / "test_module.py"
        module_path.write_text("def other_func():\n    pass\n")

        with pytest.raises(AttributeError, match="Function 'missing_func' not found"):
            import_fun_from_str(f"{module_path.with_suffix('')}.missing_func")

    def test_import_invalid_format(self):
        """Test error with invalid path format."""
        with pytest.raises(ValueError, match="Invalid path format"):
            import_fun_from_str("/path/without/function")

    def test_import_caching(self, temp_dir):
        """Test that modules are cached by mtime."""
        # Clear any existing cache
        import sys

        keys_to_remove = [k for k in sys.modules if str(temp_dir) in k]
        for k in keys_to_remove:
            del sys.modules[k]

        module_path = temp_dir / "cached_module.py"
        module_path.write_text(
            "call_count = 0\ndef func():\n    global call_count\n    call_count += 1\n    return call_count\n"
        )

        # First import
        func1 = import_fun_from_str(f"{module_path.with_suffix('')}.func")
        _result1 = func1()

        # Second import should use cached module
        func2 = import_fun_from_str(f"{module_path.with_suffix('')}.func")
        result2 = func2()

        # Both should refer to the same module instance
        assert func1 is func2
        assert result2 == 2  # call_count incremented


class TestLoad:
    """Tests for the main YAML loading function with includes and function resolution."""

    def test_load_simple_yaml(self, yaml_file):
        content = """
task: test_task
description: A test task
metric: accuracy
"""
        file_path = yaml_file(content)

        result = load_yaml(file_path)

        assert result["task"] == "test_task"
        assert result["description"] == "A test task"
        assert result["metric"] == "accuracy"

    def test_load_with_function_resolved(self, yaml_file, python_module):
        # Create a module with a function
        python_module("def process_doc(doc):\n    return doc.upper()\n")

        content = """
task: test_task
doc_to_text: !function utils.process_doc
"""
        file_path = yaml_file(content)

        result = load_yaml(file_path, resolve_functions=True)

        assert callable(result["doc_to_text"])
        assert result["doc_to_text"]("hello") == "HELLO"

    def test_load_with_function_not_resolved(self, yaml_file):
        content = """
task: test_task
doc_to_text: !function utils.process_doc
"""
        file_path = yaml_file(content)

        result = load_yaml(file_path, resolve_functions=False)

        assert isinstance(result["doc_to_text"], str)
        # When resolve_functions=False, it returns the full path + function spec
        assert result["doc_to_text"].endswith("utils.process_doc")
        assert result["doc_to_text"] == str(file_path.parent / "utils.process_doc")

    def test_load_with_includes(self, temp_dir, yaml_file):
        """Include files are merged with local values taking precedence."""
        # Create included file with shared_value: 42
        included_content = """
shared_metric: f1_score
shared_value: 42
"""
        yaml_file(included_content, "included.yaml")

        # Create main file that also defines shared_value: 100
        main_content = """
include:
  - included.yaml
task: main_task
shared_value: 100
"""
        main_path = yaml_file(main_content, "main.yaml")

        result = load_yaml(main_path, resolve_includes=True)

        assert result["task"] == "main_task"
        assert result["shared_metric"] == "f1_score"
        # Verify main file value (100) overrides included file value (42)
        assert result["shared_value"] == 100  # Local wins
        assert "include" not in result

    def test_load_with_absolute_include(self, temp_dir, yaml_file):
        # Create included file in different directory
        other_dir = temp_dir / "other"
        other_dir.mkdir()
        included_path = other_dir / "included.yaml"
        included_path.write_text("included_key: included_value\n")

        # Create main file with absolute path
        main_content = f"""
include:
  - {included_path}
main_key: main_value
"""
        main_path = yaml_file(main_content)

        result = load_yaml(main_path, resolve_includes=True)

        assert result["main_key"] == "main_value"
        assert result["included_key"] == "included_value"

    def test_load_without_includes_resolution(self, yaml_file):
        content = """
include:
  - other.yaml
task: test_task
"""
        file_path = yaml_file(content)

        result = load_yaml(file_path, resolve_includes=False)

        assert result["include"] == ["other.yaml"]
        assert result["task"] == "test_task"

    def test_load_include_cycle_detection(self, temp_dir, yaml_file):
        """Circular includes should raise ValueError."""
        # Create circular includes
        yaml_file("include:\n  - b.yaml\n", "a.yaml")
        yaml_file("include:\n  - c.yaml\n", "b.yaml")
        yaml_file("include:\n  - a.yaml\n", "c.yaml")

        with pytest.raises(ValueError, match="Include cycle"):
            load_yaml(temp_dir / "a.yaml")

    def test_load_multiple_includes(self, temp_dir, yaml_file):
        """Multiple includes are processed in order, later values override earlier."""
        # Create multiple included files
        yaml_file("key1: value1\n", "inc1.yaml")  # Sets key1 to "value1"
        yaml_file(
            "key2: value2\nmain_key: should_be_ignored\n", "inc2.yaml"
        )  # Tries to set main_key
        yaml_file(
            "key3: value3\nkey1: override\n", "inc3.yaml"
        )  # Overrides key1 to "override"

        # Include order matters: inc3 comes after inc1, so its key1 value wins
        main_content = """
include:
  - inc1.yaml
  - inc2.yaml
  - inc3.yaml
main_key: main_value
"""
        main_path = yaml_file(main_content)

        result = load_yaml(main_path)

        # Verify inc3's value overrides inc1's value for key1
        assert result["key1"] == "override"  # Last include wins
        assert result["key2"] == "value2"
        assert result["key3"] == "value3"
        # Verify main file's value is NOT overridden by inc2.yaml
        assert result["main_key"] == "main_value"  # Main file wins over includes

    def test_load_recursive_includes(self, temp_dir, yaml_file):
        """Includes can be recursive - inc1 can include inc2."""
        # Create inc2.yaml (deepest level)
        yaml_file(
            "deep_key: deep_value\nshared_key: from_inc2\nshared_middle: inc2_middle\n",
            "inc2.yaml",
        )

        # Create inc1.yaml that includes inc2.yaml
        inc1_content = """include:
  - inc2.yaml
middle_key: middle_value
shared_key: from_inc1
shared_middle: inc1_middle
"""
        yaml_file(inc1_content, "inc1.yaml")

        # Create main.yaml that includes inc1.yaml
        main_content = """include:
  - inc1.yaml
top_key: top_value
shared_key: from_main
"""
        main_path = yaml_file(main_content, "main.yaml")

        result = load_yaml(main_path)

        # All keys should be present
        assert result["deep_key"] == "deep_value"  # From inc2
        assert result["middle_key"] == "middle_value"  # From inc1
        assert result["top_key"] == "top_value"  # From main

        # Verify override order: main > inc1 > inc2
        assert result["shared_key"] == "from_main"  # Main wins
        assert result["shared_middle"] == "inc1_middle"  # inc1 wins over inc2
        assert "include" not in result  # Include directives removed

    def test_load_expanduser_path(self, yaml_file):
        """Verifies that load() calls expanduser() on paths with ~."""
        content = "test: value\n"
        file_path = yaml_file(content)

        # Mock expanduser to verify it's called and control the expansion
        with patch.object(Path, "expanduser") as mock_expand:
            mock_expand.return_value = file_path
            result = load_yaml("~/test.yaml")
            mock_expand.assert_called_once()

        assert result["test"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
