import importlib.util
import os
import py_compile
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from lm_eval.tasks._yaml_loader import _load_module_with_cache, load_yaml


BASE_NS = 1_750_000_000_000_000_000
SOURCE = (
    b"from pathlib import Path\n"
    b"marker = Path(__file__).with_suffix('.count')\n"
    b"marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')\n"
    b"def render(doc):\n"
    b"    return 'first'\n"
)


def write_helper(tmp_path, source=SOURCE, mtime=BASE_NS):
    module_path = tmp_path / "helper.py"
    module_path.write_bytes(source)
    os.utime(module_path, ns=(mtime, mtime))
    yaml_path = tmp_path / "task.yaml"
    yaml_path.write_text("doc_to_text: !function helper.render\n", encoding="utf-8")
    return module_path, yaml_path


def timestamp_bytecode(module_path):
    return Path(
        py_compile.compile(
            str(module_path),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
    )


@pytest.mark.parametrize(
    ("replacement", "delta_ns"),
    [(b"other", 100_000_000), (b"other", 1_100_000_000), (b"changed", 100_000_000)],
)
def test_yaml_helper_reloads_current_source(tmp_path, replacement, delta_ns):
    module_path, yaml_path = write_helper(tmp_path)
    bytecode = timestamp_bytecode(module_path)
    first = load_yaml(yaml_path)["doc_to_text"]
    assert first({}) == "first"
    assert bytecode.is_file()
    assert module_path.with_suffix(".count").read_text() == "1"

    updated = SOURCE.replace(b"'first'", b"'" + replacement + b"'")
    write_helper(tmp_path, updated, BASE_NS + delta_ns)
    second = load_yaml(yaml_path)["doc_to_text"]
    third = load_yaml(yaml_path)["doc_to_text"]

    assert second({}) == replacement.decode()
    assert third({}) == replacement.decode()
    assert second is not first
    assert third is second
    assert module_path.with_suffix(".count").read_text() == "2"


def test_yaml_helper_ignores_stale_bytecode_on_first_load(tmp_path):
    module_path, yaml_path = write_helper(tmp_path)
    bytecode = timestamp_bytecode(module_path)
    updated = SOURCE.replace(b"'first'", b"'other'")
    assert len(updated) == len(SOURCE)
    write_helper(tmp_path, updated, BASE_NS + 100_000_000)
    assert bytecode.is_file()

    render = load_yaml(yaml_path)["doc_to_text"]
    assert render({}) == "other"
    assert module_path.with_suffix(".count").read_text() == "1"


def test_unchanged_yaml_helper_reuses_module_without_execution(tmp_path):
    module_path, yaml_path = write_helper(tmp_path)
    timestamp_bytecode(module_path)
    first = load_yaml(yaml_path)["doc_to_text"]
    module = sys.modules[first.__module__]

    assert load_yaml(yaml_path)["doc_to_text"] is first
    assert sys.modules[first.__module__] is module
    assert module_path.with_suffix(".count").read_text() == "1"


@pytest.mark.parametrize("syntax_error", [False, True])
def test_failed_yaml_helper_reload_keeps_last_good_module(tmp_path, syntax_error):
    module_path, yaml_path = write_helper(tmp_path)
    timestamp_bytecode(module_path)
    first = load_yaml(yaml_path)["doc_to_text"]
    module = sys.modules[first.__module__]
    if syntax_error:
        updated = SOURCE.replace(b"def render(doc):", b"def render(doc);")
        assert len(updated) == len(SOURCE)
        expected_error = SyntaxError
    else:
        updated = b"raise RuntimeError('broken helper')\n" + SOURCE
        expected_error = RuntimeError
    write_helper(tmp_path, updated, BASE_NS + 100_000_000)

    for _attempt in range(2):
        with pytest.raises(expected_error):
            load_yaml(yaml_path)
        assert sys.modules[first.__module__] is module
        assert first({}) == "first"


@pytest.mark.parametrize("future_annotations", [False, True])
def test_yaml_helper_preserves_own_future_flags(tmp_path, future_annotations):
    source = b"def render(doc: MissingType):\n    return 'value'\n"
    if future_annotations:
        source = b"from __future__ import annotations\n" + source
    _module_path, yaml_path = write_helper(tmp_path, source)

    if future_annotations:
        render = load_yaml(yaml_path)["doc_to_text"]
        assert render.__annotations__ == {"doc": "MissingType"}
        assert render({}) == "value"
    else:
        with pytest.raises(NameError, match="MissingType"):
            dict(load_yaml(yaml_path)["doc_to_text"].__annotations__)


@pytest.mark.parametrize(
    "source",
    [
        "# coding: latin-1\ndef render(doc):\n    return 'café'\n".encode("latin-1"),
        "def render(doc):\n    return 'café'\n".encode("utf-8-sig"),
    ],
)
def test_yaml_helper_preserves_python_source_encodings(tmp_path, source):
    _module_path, yaml_path = write_helper(tmp_path, source)
    assert load_yaml(yaml_path)["doc_to_text"]({}) == "café"


def test_yaml_helper_keeps_source_module_metadata(tmp_path):
    module_path, yaml_path = write_helper(tmp_path)
    render = load_yaml(yaml_path)["doc_to_text"]
    module = sys.modules[render.__module__]

    assert module.__file__ == str(module_path)
    assert module.__name__ == render.__module__
    assert module.__spec__.origin == str(module_path)
    assert module.__spec__.name == module.__name__
    assert module.__spec__.loader is module.__loader__
    assert isinstance(module.__loader__, SourceFileLoader)
    assert module.__package__ == module.__spec__.parent
    assert module.__cached__ == importlib.util.cache_from_source(str(module_path))
    assert render.__code__.co_filename == str(module_path)


def test_unsupported_module_suffix_still_fails(tmp_path):
    module_path = tmp_path / "helper.unknown"
    module_path.write_bytes(SOURCE)
    with pytest.raises(ImportError, match="Cannot load module"):
        _load_module_with_cache(module_path)


def test_relative_helper_path_keeps_absolute_module_metadata(tmp_path, monkeypatch):
    module_path, _yaml_path = write_helper(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "helper", None)
    module = _load_module_with_cache(Path("helper.py"))

    assert module.__file__ == str(module_path)
    assert module.__spec__.origin == str(module_path)
    assert module.__loader__.get_filename(module.__name__) == str(module_path)
    assert module.render.__code__.co_filename == str(module_path)


def test_sourceless_module_loading_is_unchanged(tmp_path):
    module_path, _yaml_path = write_helper(tmp_path)
    bytecode = timestamp_bytecode(module_path)
    module = _load_module_with_cache(bytecode)
    assert module.render({}) == "first"
