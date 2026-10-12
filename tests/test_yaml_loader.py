"""Behavioral tests for local YAML function imports."""

import importlib
import json
import os
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from lm_eval.api.registry import import_plugins
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import _load_module_with_cache, load_yaml


@pytest.fixture(autouse=True)
def clean_local_modules(tmp_path):
    """Keep dynamically imported test packages out of subsequent tests."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        filename = getattr(module, "__file__", None)
        if filename and str(tmp_path) in filename:
            sys.modules.pop(name, None)


def write_files(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def load_function(root):
    before = sys.path.copy()
    try:
        return load_yaml(root / "task.yaml")["doc_to_text"]
    finally:
        assert sys.path == before


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_package_relative_imports(tmp_path, nested):
    """Missing package context breaks the original registry and nested imports."""
    root = tmp_path / "custom_tasks"
    write_files(
        root,
        {
            "__init__.py": "PREFIX = 'package:'\n",
            "instructions_registry.py": "VALUE = 'registry'\n",
            "nested/__init__.py": "",
            "nested/helper.py": "from .. import instructions_registry\n"
            "VALUE = instructions_registry.VALUE + ':nested'\n",
            "utils.py": (
                "from .nested.helper import VALUE\n"
                if nested
                else "from . import instructions_registry\nVALUE = instructions_registry.VALUE\n"
            )
            + "from . import PREFIX\ndef render(doc):\n    return PREFIX + VALUE\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    render = load_function(root)
    assert render({}) == ("package:registry:nested" if nested else "package:registry")


@pytest.mark.parametrize("dirname", ["custom_tasks", "tasks-with-dashes.v1", "json"])
def test_same_named_packages_are_isolated(tmp_path, dirname):
    """Root directory spelling and unrelated installed names cannot alias paths."""
    roots = [tmp_path / str(i) / dirname for i in range(2)]
    for root, value in zip(roots, [17, 29], strict=True):
        write_files(
            root,
            {
                "__init__.py": "",
                "helper.py": f"VALUE = {value}\n",
                "utils.py": "from .helper import VALUE\ndef render(doc):\n    return VALUE\n",
                "task.yaml": "doc_to_text: !function utils.render\n",
            },
        )
    first, second = [load_function(root) for root in roots]
    assert [first({}), second({})] == [17, 29]
    assert load_function(roots[1]) is second
    assert load_function(roots[0]) is first
    assert first.__module__ != second.__module__
    assert sys.modules["json"] is json
    assert load_function(roots[0] / ".." / dirname) is first
    if os.name == "nt":
        assert load_function(Path(str(roots[0]).upper())) is first


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("preimport", [False, True])
def test_package_initialization_imports_requested_module_once(
    tmp_path, monkeypatch, installed, preimport
):
    """An initializer's child import must be the same module returned by YAML."""
    root = tmp_path / "init_task_package"
    write_files(
        root,
        {
            "__init__.py": "EVENTS = ['init']\nfrom . import utils\nEVENTS.append('ready')\n",
            "utils.py": "from . import EVENTS\nEVENTS.append('utils')\n"
            "def render(doc):\n    return EVENTS\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    if installed:
        monkeypatch.syspath_prepend(str(tmp_path))
        if preimport:
            import_plugins(["init_task_package"])
    render = load_function(root)
    assert render({}) == ["init", "utils", "ready"]
    assert load_function(root) is render
    module = sys.modules[render.__module__]
    package = sys.modules[module.__package__]
    assert package.utils is module
    if installed:
        assert render is importlib.import_module("init_task_package.utils").render
        if os.name == "nt":
            assert load_function(Path(str(root).upper())) is render


def test_circular_relative_imports_and_concurrent_loads(tmp_path):
    """The requested child must be registered before its helper imports it."""
    write_files(
        tmp_path,
        {
            "__init__.py": "",
            "utils.py": "VALUE = 31\nfrom .helper import RESULT\n"
            "def render(doc):\n    return RESULT\n",
            "helper.py": "from .utils import VALUE\nRESULT = VALUE + 1\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        functions = list(pool.map(lambda _: load_function(tmp_path), range(8)))
    assert all(fn is functions[0] for fn in functions)
    assert functions[0]({}) == 32


@pytest.mark.parametrize(
    "failure_file", ["__init__.py", "nested/__init__.py", "utils.py"]
)
@pytest.mark.parametrize("installed", [False, True])
def test_failed_import_removes_new_modules_and_allows_retry(
    tmp_path, monkeypatch, failure_file, installed
):
    """A failed transaction must not leave successful helpers cached."""
    files = {
        "__init__.py": "from . import helper\n",
        "helper.py": "VALUE = 41\n",
        "nested/__init__.py": "from .. import helper\n",
        "nested/child.py": "from ..helper import VALUE\n",
        "utils.py": "from .nested.child import VALUE\ndef render(doc):\n    return VALUE\n",
        "task.yaml": "doc_to_text: !function utils.render\n",
    }
    write_files(tmp_path, files)
    if installed:
        monkeypatch.syspath_prepend(str(tmp_path.parent))
    failed = tmp_path / failure_file
    failed.write_text(files[failure_file] + "raise RuntimeError('broken task')\n")
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="broken task"):
        load_function(tmp_path)
    leaked = [
        name
        for name in set(sys.modules) - before
        if str(tmp_path) in str(getattr(sys.modules[name], "__file__", ""))
    ]
    assert leaked == []
    failed.write_text(files[failure_file])
    assert load_function(tmp_path)({}) == 41


def replace_source(path, source):
    old_mtime = path.stat().st_mtime_ns
    path.write_text(source, encoding="utf-8")
    # Cross a bytecode-cache timestamp boundary without sleeping.
    os.utime(path, ns=(old_mtime + 3_000_000_000, old_mtime + 3_000_000_000))


def test_failed_reload_preserves_existing_modules_and_parent_bindings(tmp_path):
    """A reload error cannot destroy the last usable module or keep new helpers."""
    write_files(
        tmp_path,
        {
            "__init__.py": "",
            "helper.py": "VALUE = 5\n",
            "utils.py": "from .helper import VALUE\ndef render(doc):\n    return VALUE\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    original = load_function(tmp_path)
    module = sys.modules[original.__module__]
    package = sys.modules[module.__package__]
    helper = package.helper
    (tmp_path / "newhelper.py").write_text("VALUE = 9\n")
    replace_source(
        tmp_path / "utils.py",
        "from . import newhelper\nraise RuntimeError('broken reload')\n",
    )
    with pytest.raises(RuntimeError, match="broken reload"):
        load_function(tmp_path)
    assert sys.modules[original.__module__] is module
    assert package.utils is module
    assert package.helper is helper
    assert not hasattr(package, "newhelper")
    assert f"{module.__package__}.newhelper" not in sys.modules
    assert original({}) == 5
    replace_source(
        tmp_path / "utils.py",
        "from .helper import VALUE\ndef render(doc):\n    return VALUE + 2\n",
    )
    assert load_function(tmp_path)({}) == 7


@pytest.mark.parametrize("packaged", [False, True])
def test_main_reload_and_helper_cache(tmp_path, packaged):
    """Repeat loads reuse functions; changing only a helper does not reload it."""
    source = "def render(doc):\n    return 5\n"
    if packaged:
        write_files(tmp_path, {"__init__.py": "", "helper.py": "VALUE = 5\n"})
        source = "from .helper import VALUE\ndef render(doc):\n    return VALUE\n"
    write_files(
        tmp_path,
        {"utils.py": source, "task.yaml": "doc_to_text: !function utils.render\n"},
    )
    original = load_function(tmp_path)
    assert original({}) == 5
    assert load_function(tmp_path) is original
    if packaged:
        replace_source(tmp_path / "helper.py", "VALUE = 100\n")
        assert load_function(tmp_path) is original
    replace_source(tmp_path / "utils.py", source.replace("return ", "return 2 + "))
    changed = load_function(tmp_path)
    assert changed is not original
    assert changed({}) == 7
    assert load_function(tmp_path) is changed


@pytest.mark.parametrize("installed", [False, True])
def test_local_file_takes_precedence_over_same_named_package(
    tmp_path, monkeypatch, installed
):
    """YAML local-file resolution must not switch to importlib's package priority."""
    write_files(
        tmp_path,
        {
            "__init__.py": "from . import utils\n",
            "utils.py": "def render(doc):\n    return 'file'\n",
            "utils/__init__.py": "def render(doc):\n    return 'package'\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    if installed:
        monkeypatch.syspath_prepend(str(tmp_path.parent))
        package = importlib.import_module(tmp_path.name)
        original = package.utils
    render = load_function(tmp_path)
    assert render({}) == "file"
    assert load_function(tmp_path) is render
    parent = sys.modules[sys.modules[render.__module__].__package__]
    assert parent.utils.render({}) == "package"
    if installed:
        assert package.utils is original
        assert importlib.import_module(tmp_path.name + ".utils") is original


def test_reload_publishes_new_parent_binding_before_helper_import(tmp_path):
    """A newly imported helper must see the replacement module through its parent."""
    write_files(
        tmp_path,
        {
            "__init__.py": "",
            "utils.py": "VALUE = 1\ndef render(doc):\n    return VALUE\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    assert load_function(tmp_path)({}) == 1
    (tmp_path / "helper.py").write_text("from . import utils\nVALUE = utils.VALUE\n")
    replace_source(
        tmp_path / "utils.py",
        "VALUE = 2\nfrom .helper import VALUE as RESULT\ndef render(doc):\n    return RESULT\n",
    )
    assert load_function(tmp_path)({}) == 2


def test_reload_discovers_new_helper_with_unchanged_directory_mtime(tmp_path):
    """A finder cached before file creation must discover new reload dependencies."""
    write_files(
        tmp_path,
        {
            "__init__.py": "",
            "helper.py": "VALUE = 5\n",
            "utils.py": "from .helper import VALUE\ndef render(doc):\n    return VALUE\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    original = load_function(tmp_path)
    package = sys.modules[sys.modules[original.__module__].__package__]
    helper = package.helper
    directory_mtime = tmp_path.stat().st_mtime_ns
    (tmp_path / "newhelper.py").write_text("VALUE = 63\n")
    replace_source(
        tmp_path / "utils.py",
        "from .newhelper import VALUE\ndef render(doc):\n    return VALUE\n",
    )
    # Reproduce files created within one directory timestamp resolution interval.
    os.utime(tmp_path, ns=(directory_mtime, directory_mtime))
    assert load_function(tmp_path)({}) == 63
    assert package.helper is helper


@pytest.mark.parametrize("standard_first", [True, False])
def test_yaml_and_standard_import_wait_for_initialization(
    tmp_path, monkeypatch, standard_first
):
    """Plugin imports and YAML loads must not expose one another's partial modules."""
    root = tmp_path / "concurrent_task_package"
    sync = types.ModuleType("yaml_loader_test_sync")
    sync.started = threading.Event()
    sync.release = threading.Event()
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    monkeypatch.syspath_prepend(str(tmp_path))
    write_files(
        root,
        {
            "__init__.py": "",
            "utils.py": "from yaml_loader_test_sync import started, release\n"
            "started.set()\nassert release.wait(10)\ndef render(doc):\n    return 23\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    importlib.import_module("concurrent_task_package")
    with ThreadPoolExecutor(max_workers=2) as pool:
        if standard_first:
            first = pool.submit(
                importlib.import_module, "concurrent_task_package.utils"
            )
        else:
            first = pool.submit(load_function, root)
        assert sync.started.wait(10)
        if standard_first:
            second = pool.submit(load_function, root)
        else:
            second = pool.submit(
                importlib.import_module, "concurrent_task_package.utils"
            )
        timer = threading.Timer(0.2, sync.release.set)
        timer.start()
        try:
            left, right = first.result(timeout=10), second.result(timeout=10)
        finally:
            sync.release.set()
            timer.join()
    render = right if standard_first else left
    module = left if standard_first else right
    assert render({}) == 23
    assert render is module.render


def test_package_initializer_can_load_yaml_during_another_request(tmp_path):
    """A loader lock must not invert the standard root-import lock order."""
    write_files(
        tmp_path / "root_lock_package",
        {
            "__init__.py": "from root_lock_sync import entered, requested, load\n"
            "entered.set()\nassert requested.wait(5)\nRESULT = load()({})\n",
            "utils.py": "def render(doc):\n    return 37\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    # Isolate a potential deadlock so the pytest process never retains stuck threads.
    script = """
import importlib, sys, threading, types
from pathlib import Path
from lm_eval.tasks import _yaml_loader as loader
root = Path(sys.argv[1]) / "root_lock_package"
sys.path.insert(0, str(root.parent))
sync = types.ModuleType("root_lock_sync")
sync.entered = threading.Event()
sync.requested = threading.Event()
sync.load = lambda: loader.load_yaml(root / "task.yaml")["doc_to_text"]
sys.modules[sync.__name__] = sync
original = loader._package_name
def observe(path):
    if threading.current_thread().name == "yaml-request":
        sync.requested.set()
    return original(path)
loader._package_name = observe
results = []
errors = []
def call(action):
    try:
        results.append(action())
    except BaseException as exc:
        errors.append(repr(exc))
standard = threading.Thread(target=lambda: call(lambda: importlib.import_module("root_lock_package").RESULT), daemon=True)
request = threading.Thread(name="yaml-request", target=lambda: call(lambda: sync.load()({})), daemon=True)
standard.start()
assert sync.entered.wait(5)
request.start()
standard.join(6)
request.join(6)
assert not standard.is_alive() and not request.is_alive(), "import/loader lock inversion"
assert not errors, errors
assert results == [37, 37], results
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_standard_import_waits_for_yaml_reload(tmp_path, monkeypatch):
    """Normal imports must wait for file execution during mtime replacement too."""
    root = tmp_path / "reload_import_package"
    sync = types.ModuleType("yaml_reload_test_sync")
    sync.started = threading.Event()
    sync.release = threading.Event()
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    monkeypatch.syspath_prepend(str(tmp_path))
    write_files(
        root,
        {
            "__init__.py": "",
            "utils.py": "def render(doc):\n    return 1\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    assert load_function(root)({}) == 1
    replace_source(
        root / "utils.py",
        "from yaml_reload_test_sync import started, release\n"
        "started.set()\nassert release.wait(10)\ndef render(doc):\n    return 27\n",
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        loaded = pool.submit(load_function, root)
        assert sync.started.wait(10)
        imported = pool.submit(
            lambda: importlib.import_module("reload_import_package.utils").render({})
        )
        timer = threading.Timer(0.2, sync.release.set)
        timer.start()
        try:
            render = loaded.result(timeout=10)
            result = imported.result(timeout=10)
        finally:
            sync.release.set()
            timer.join()
    assert render({}) == result == 27


def test_fromlist_waits_for_yaml_reload(tmp_path, monkeypatch):
    """A parent attribute must not bypass synchronization for from-list imports."""
    root = tmp_path / "fromlist_reload_package"
    sync = types.ModuleType("fromlist_reload_sync")
    sync.started = threading.Event()
    sync.release = threading.Event()
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    monkeypatch.syspath_prepend(str(tmp_path))
    write_files(
        root,
        {
            "__init__.py": "",
            "utils.py": "def render(doc):\n    return 1\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    assert load_function(root)({}) == 1
    replace_source(
        root / "utils.py",
        "from fromlist_reload_sync import started, release\n"
        "started.set()\nassert release.wait(10)\ndef render(doc):\n    return 29\n",
    )

    def fromlist():
        from fromlist_reload_package import utils

        return utils.render({})

    with ThreadPoolExecutor(max_workers=2) as pool:
        loaded = pool.submit(load_function, root)
        assert sync.started.wait(10)
        imported = pool.submit(fromlist)
        timer = threading.Timer(0.2, sync.release.set)
        timer.start()
        try:
            render = loaded.result(timeout=10)
            result = imported.result(timeout=10)
        finally:
            sync.release.set()
            timer.join()
    assert render({}) == result == 29


def test_standard_child_initializer_can_load_another_yaml(tmp_path, monkeypatch):
    """Waiting for utils must not lock its initialized root against YAML callbacks."""
    from lm_eval.tasks import _yaml_loader as loader

    root = tmp_path / "callback_import_package"
    sync = types.ModuleType("child_callback_sync")
    sync.started = threading.Event()
    sync.requested = threading.Event()
    sync.load = lambda: load_yaml(root / "helper.yaml")["doc_to_text"]({})
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    monkeypatch.syspath_prepend(str(tmp_path))
    write_files(
        root,
        {
            "__init__.py": "",
            "utils.py": "from child_callback_sync import started, requested, load\n"
            "started.set()\nassert requested.wait(10)\nRESULT = load()\n"
            "def render(doc):\n    return RESULT\n",
            "helper.py": "def render(doc):\n    return 41\n",
            "helper.yaml": "doc_to_text: !function helper.render\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    importlib.import_module("callback_import_package")
    original = loader._cached_module

    def observe(name, path):
        if name == "callback_import_package.utils":
            sync.requested.set()
        return original(name, path)

    monkeypatch.setattr(loader, "_cached_module", observe)
    with ThreadPoolExecutor(max_workers=2) as pool:
        imported = pool.submit(
            lambda: importlib.import_module("callback_import_package.utils").render({})
        )
        assert sync.started.wait(10)
        loaded = pool.submit(load_function, root)
        assert imported.result(timeout=10) == 41
        assert loaded.result(timeout=10)({}) == 41


@pytest.mark.parametrize("other_succeeds", [False, True])
def test_overlapping_package_loads_preserve_success_or_rollback(
    tmp_path, monkeypatch, other_succeeds
):
    """A failed request must not remove another request's still executing module."""
    sync = types.ModuleType("overlap_load_sync")
    sync.barrier = threading.Barrier(3)
    sync.release = threading.Event()
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    write_files(
        tmp_path,
        {
            "__init__.py": "",
            "helper.py": "VALUE = 43\n",
            "left.py": "from .helper import VALUE\nfrom overlap_load_sync import barrier\n"
            "barrier.wait(10)\nraise RuntimeError('left failed')\n",
            "right.py": "from .helper import VALUE\nfrom overlap_load_sync import barrier, release\n"
            "barrier.wait(10)\nassert release.wait(10)\n"
            + (
                "def render(doc):\n    return VALUE\n"
                if other_succeeds
                else "raise RuntimeError('right failed')\n"
            ),
            "left.yaml": "doc_to_text: !function left.render\n",
            "right.yaml": "doc_to_text: !function right.render\n",
        },
    )
    before = set(sys.modules)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(load_yaml, tmp_path / "left.yaml")
        right = pool.submit(load_yaml, tmp_path / "right.yaml")
        try:
            sync.barrier.wait(10)
            with pytest.raises(RuntimeError, match="left failed"):
                left.result(timeout=10)
        finally:
            sync.release.set()
        if other_succeeds:
            render = right.result(timeout=10)["doc_to_text"]
            assert render({}) == 43
            assert load_yaml(tmp_path / "right.yaml")["doc_to_text"] is render
        else:
            with pytest.raises(RuntimeError, match="right failed"):
                right.result(timeout=10)
    if not other_succeeds:
        assert not [
            name
            for name in set(sys.modules) - before
            if str(tmp_path) in str(getattr(sys.modules[name], "__file__", ""))
        ]


def test_initializer_failure_rolls_back_successful_nested_yaml_load(
    tmp_path, monkeypatch
):
    """A nested YAML success does not commit a subsequently failing initializer."""
    sync = types.ModuleType("nested_initializer_sync")
    sync.load = lambda: load_yaml(tmp_path / "helper.yaml")["doc_to_text"]({})
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    write_files(
        tmp_path,
        {
            "__init__.py": "from nested_initializer_sync import load\n"
            "assert load() == 47\nraise RuntimeError('initializer failed')\n",
            "helper.py": "def render(doc):\n    return 47\n",
            "helper.yaml": "doc_to_text: !function helper.render\n",
            "utils.py": "def render(doc):\n    return 48\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="initializer failed"):
        load_function(tmp_path)
    assert not [
        name
        for name in set(sys.modules) - before
        if str(tmp_path) in str(getattr(sys.modules[name], "__file__", ""))
    ]


@pytest.mark.parametrize("failed_module", ["__init__.py", "utils.py"])
def test_failed_standard_initialization_is_not_restored(
    tmp_path, monkeypatch, failed_module
):
    """Rollback must not resurrect a standard import's failed partial snapshot."""
    from lm_eval.tasks import _yaml_loader as loader

    root = tmp_path / "failed_snapshot_package"
    sync = types.ModuleType("failed_snapshot_sync")
    sync.started = threading.Event()
    sync.requested = threading.Event()
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    monkeypatch.syspath_prepend(str(tmp_path))
    write_files(
        root,
        {
            "__init__.py": "FLAG = 53\n",
            "utils.py": "def render(doc):\n    return 1\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    (root / failed_module).write_text(
        "from failed_snapshot_sync import started, requested\n"
        "started.set()\nassert requested.wait(10)\nraise RuntimeError('failed initialization')\n"
    )
    name = "failed_snapshot_package"
    if failed_module == "utils.py":
        package = importlib.import_module(name)
        name += ".utils"
    original = loader._package_transaction

    @contextmanager
    def observe(package_name):
        with original(package_name) as state:
            sync.requested.set()
            yield state

    monkeypatch.setattr(loader, "_package_transaction", observe)
    with ThreadPoolExecutor(max_workers=2) as pool:
        imported = pool.submit(importlib.import_module, name)
        assert sync.started.wait(10)
        loaded = pool.submit(load_function, root)
        with pytest.raises(RuntimeError, match="failed initialization"):
            imported.result(timeout=10)
        with pytest.raises(RuntimeError, match="failed initialization"):
            loaded.result(timeout=10)
    assert name not in sys.modules
    with pytest.raises(RuntimeError, match="failed initialization"):
        load_function(root)
    if failed_module == "utils.py":
        assert sys.modules["failed_snapshot_package"] is package
        assert package.FLAG == 53


@pytest.mark.parametrize("initializing_child", [False, True])
def test_completed_standard_dependencies_survive_yaml_failure(
    tmp_path, monkeypatch, initializing_child
):
    """After waiting for initialization, completed standard imports are preexisting."""
    from lm_eval.tasks import _yaml_loader as loader

    root = tmp_path / "completed_snapshot_package"
    sync = types.ModuleType("completed_snapshot_sync")
    sync.started = threading.Event()
    sync.requested = threading.Event()
    monkeypatch.setitem(sys.modules, sync.__name__, sync)
    monkeypatch.syspath_prepend(str(tmp_path))
    write_files(
        root,
        {
            "__init__.py": "",
            "helper.py": "from . import dep\nINITIAL_DEP = dep\n"
            "def check():\n    from . import dep\n    return dep is INITIAL_DEP\n",
            "dep.py": "VALUE = 59\n",
            "utils.py": "from .helper import check\nassert check()\n"
            "raise RuntimeError('utils failed')\n",
            "task.yaml": "doc_to_text: !function utils.render\n",
        },
    )
    name = "completed_snapshot_package"
    initializing_file = root / "__init__.py"
    initialization = "from . import helper\n"
    if initializing_child:
        importlib.import_module(name)
        name += ".helper"
        initializing_file = root / "helper.py"
        initialization = initializing_file.read_text()
    initializing_file.write_text(
        "from completed_snapshot_sync import started, requested\n"
        "started.set()\nassert requested.wait(10)\n" + initialization
    )
    original = loader._package_transaction

    @contextmanager
    def observe(package_name):
        with original(package_name) as state:
            sync.requested.set()
            yield state

    monkeypatch.setattr(loader, "_package_transaction", observe)
    with ThreadPoolExecutor(max_workers=2) as pool:
        imported = pool.submit(importlib.import_module, name)
        assert sync.started.wait(10)
        loaded = pool.submit(load_function, root)
        imported.result(timeout=10)
        package = sys.modules["completed_snapshot_package"]
        helper = package.helper
        with pytest.raises(RuntimeError, match="utils failed"):
            loaded.result(timeout=10)
    assert sys.modules["completed_snapshot_package"] is package
    assert sys.modules["completed_snapshot_package.helper"] is helper
    assert package.helper is helper
    assert sys.modules["completed_snapshot_package.dep"] is helper.INITIAL_DEP
    assert package.dep is helper.INITIAL_DEP
    assert helper.check()


def test_absolute_module_reference(tmp_path):
    write_files(tmp_path, {"task.yaml": "doc_to_text: !function json.dumps\n"})
    assert load_function(tmp_path) is json.dumps


def test_nested_function_path_initializes_parent_before_child(tmp_path):
    """A dotted YAML module path must initialize its intermediate package once."""
    write_files(
        tmp_path,
        {
            "__init__.py": "EVENTS = ['root']\n",
            "nested/__init__.py": "from .. import EVENTS\nEVENTS.append('nested')\nfrom . import utils\n",
            "nested/utils.py": "from .. import EVENTS\nEVENTS.append('utils')\n"
            "def render(doc):\n    return EVENTS\n",
            "task.yaml": "doc_to_text: !function nested.utils.render\n",
        },
    )
    render = load_function(tmp_path)
    assert render({}) == ["root", "nested", "utils"]
    assert load_function(tmp_path) is render


def test_builtin_module_uses_canonical_name():
    """Windows native paths must preserve built-in module identity."""
    import lm_eval.tasks._yaml_loader as loader

    module = _load_module_with_cache(Path(loader.__file__).parent / "_yaml_loader.py")
    assert module.__name__ == "lm_eval.tasks._yaml_loader"


def test_builtin_yaml_function_result(tmp_path):
    """An actual built-in function keeps its canonical identity and behavior."""
    import lm_eval.tasks._yaml_loader as loader

    task_dir = Path(loader.__file__).parent / "hellaswag"
    write_files(
        tmp_path,
        {"task.yaml": f"include: {json.dumps(str(task_dir / 'hellaswag.yaml'))}\n"},
    )
    config = load_yaml(tmp_path / "task.yaml")
    from datasets import Dataset

    docs = Dataset.from_list(
        [
            {
                "ctx_a": "Start",
                "ctx_b": "next",
                "activity_label": "Test",
                "endings": ["one", "two"],
                "label": "1",
            }
        ]
    )
    result = config["process_docs"](docs)
    assert result.select_columns(["query", "choices", "gold"])[0] == {
        "query": "Test: Start Next",
        "choices": ["one", "two"],
        "gold": 1,
    }
    assert config["process_docs"].__module__ == "lm_eval.tasks.hellaswag.utils"


def test_task_manager_with_package_and_local_dataset(tmp_path):
    """Actual task discovery, YAML resolution and local dataset construction."""
    root = tmp_path / "local_tasks"
    write_files(
        root,
        {
            "__init__.py": "",
            "instructions_registry.py": "PREFIX = 'Question: '\n",
            "utils.py": "from . import instructions_registry\n"
            "def render(doc):\n    return instructions_registry.PREFIX + doc['question']\n",
            "data.jsonl": '{"question":"2+2?","answer":"4"}\n',
        },
    )
    write_files(
        root,
        {
            "task.yaml": (
                "task: package_local_test\ndataset_path: json\ndataset_kwargs:\n"
                f"  data_files:\n    test: {json.dumps(str(root / 'data.jsonl'))}\n"
                "test_split: test\noutput_type: generate_until\n"
                "doc_to_text: !function utils.render\ndoc_to_target: answer\n"
                "generation_kwargs:\n  until: []\nmetric_list:\n  - metric: exact_match\n"
            )
        },
    )
    manager = TaskManager(include_path=root, include_defaults=False)
    task = manager.load("package_local_test")["tasks"]["package_local_test"]
    docs = list(task.test_docs())
    assert len(docs) == 1
    assert task.doc_to_text(docs[0]) == "Question: 2+2?"
    assert task.doc_to_target(docs[0]) == "4"
