"""Tests for lm_eval._include and the --include_path / --include_module CLI flags."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

import lm_eval.models  # noqa: F401 — load built-ins before registering fakes
from lm_eval import _include
from lm_eval.api.registry import get_model, model_registry


_USER_LM_SRC = textwrap.dedent(
    """
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model


    @register_model("{alias}")
    class FakeUserLM(LM):
        def __init__(self, **kwargs):
            super().__init__()

        def loglikelihood(self, requests):
            return []

        def loglikelihood_rolling(self, requests):
            return []

        def generate_until(self, requests):
            return []
    """
)


@pytest.fixture(autouse=True)
def _reset_include_caches():
    _include._imported_paths.clear()
    _include._imported_modules.clear()
    try:
        yield
    finally:
        _include._imported_paths.clear()
        _include._imported_modules.clear()
        for name in list(sys.modules):
            if name.startswith(("lm_eval_user_", "issue1457_pkg")):
                sys.modules.pop(name, None)


def _write_user_lm(tmp_path: Path, alias: str, filename: str = "my_lm.py") -> Path:
    f = tmp_path / filename
    f.write_text(_USER_LM_SRC.format(alias=alias))
    return f


def test_include_path_single_file(tmp_path: Path):
    alias = "issue1457-from-file"
    py_file = _write_user_lm(tmp_path, alias)

    assert alias not in model_registry
    _include.import_user_modules(include_path=py_file)
    assert alias in model_registry
    assert get_model(alias).__name__ == "FakeUserLM"


def test_include_path_directory_is_not_auto_imported(tmp_path: Path):
    alias = "issue1457-dir-must-stay-lazy"
    _write_user_lm(tmp_path, alias)

    _include.import_user_modules(include_path=tmp_path)
    assert alias not in model_registry


def test_include_path_does_not_mutate_sys_path_or_shadow_modules(tmp_path: Path):
    (tmp_path / "packaging.py").write_text("SENTINEL = 'user-decoy'")
    alias = "issue1457-no-global-leak"
    py_file = _write_user_lm(tmp_path, alias)

    sys_path_before = list(sys.path)
    _include.import_user_modules(include_path=py_file)

    assert list(sys.path) == sys_path_before
    assert str(tmp_path) not in sys.path

    for name, mod in list(sys.modules.items()):
        if name.startswith("lm_eval_user_"):
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            resolved = str(Path(f).resolve())
        except OSError:
            continue
        assert str(tmp_path.resolve()) not in resolved, (
            f"sys.modules['{name}'] was loaded from user dir {f}"
        )


def test_sibling_import_fails_cleanly(tmp_path: Path):
    (tmp_path / "helpers.py").write_text("SENTINEL = 'ok'")
    main = tmp_path / "main.py"
    main.write_text("from helpers import SENTINEL  # noqa: F401\n")

    with pytest.raises(ImportError) as exc_info:
        _include.import_user_modules(include_path=main)

    msg = str(exc_info.value)
    assert "main.py" in msg
    assert "--include_module" in msg


def test_reimport_is_noop(tmp_path: Path):
    alias = "issue1457-reimport"
    py_file = _write_user_lm(tmp_path, alias)

    _include.import_user_modules(include_path=py_file)
    _include.import_user_modules(include_path=py_file)
    assert alias in model_registry


def test_broken_user_file_raises_with_path(tmp_path: Path):
    bad = tmp_path / "broken.py"
    bad.write_text("this is not valid python :::")

    with pytest.raises(ImportError) as exc_info:
        _include.import_user_modules(include_path=bad)

    assert "broken.py" in str(exc_info.value)


def test_include_module_dotted_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "issue1457_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "lms.py").write_text(_USER_LM_SRC.format(alias="issue1457-from-module"))

    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name.startswith("issue1457_pkg"):
            sys.modules.pop(name)

    _include.import_user_modules(include_module="issue1457_pkg.lms")
    assert "issue1457-from-module" in model_registry


def test_nonexistent_include_path_is_ignored(tmp_path: Path):
    _include.import_user_modules(include_path=tmp_path / "does-not-exist")


def test_non_py_existing_file_warns_and_skips(tmp_path, caplog):
    odd = tmp_path / "my_lm.pyc"
    odd.write_text("not python")

    with caplog.at_level("WARNING", logger="lm_eval._include"):
        _include.import_user_modules(include_path=odd)

    assert any("my_lm.pyc" in r.getMessage() for r in caplog.records)


def test_module_then_same_file_no_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pkg = tmp_path / "issue1457_pkg"
    pkg.mkdir()
    alias = "issue1457-module-then-file"
    (pkg / "__init__.py").write_text(_USER_LM_SRC.format(alias=alias))

    monkeypatch.syspath_prepend(str(tmp_path))
    _include.import_user_modules(include_module="issue1457_pkg")
    assert alias in model_registry

    _include.import_user_modules(include_path=pkg / "__init__.py")
    assert alias in model_registry


def test_configure_triggers_import_for_py_file(tmp_path: Path):
    from lm_eval.config.evaluate_config import EvaluatorConfig

    alias = "issue1457-via-config"
    py_file = _write_user_lm(tmp_path, alias)

    EvaluatorConfig(
        model="dummy",
        tasks=["hellaswag"],
        include_path=str(py_file),
    )._configure()

    assert alias in model_registry
    assert get_model(alias).__name__ == "FakeUserLM"


def test_task_discovery_path_splits_py_file_from_yaml_root():
    assert _include.task_discovery_path(None) is None
    assert _include.task_discovery_path("./some_dir") == "./some_dir"
    assert _include.task_discovery_path(Path(__file__)) == str(Path(__file__).parent)


def test_configure_does_not_import_directory_py_files(tmp_path: Path):
    from lm_eval.config.evaluate_config import EvaluatorConfig

    alias = "issue1457-dir-via-config"
    _write_user_lm(tmp_path, alias)

    EvaluatorConfig(
        model="dummy",
        tasks=["hellaswag"],
        include_path=str(tmp_path),
    )._configure()

    assert alias not in model_registry


def test_register_metric_and_filter_via_include_path(tmp_path: Path):
    from lm_eval.api.registry import (
        filter_registry,
        get_aggregation,
        metric_registry,
    )

    py_file = tmp_path / "extras.py"
    py_file.write_text(
        textwrap.dedent(
            """
            from lm_eval.api.filter import Filter
            from lm_eval.api.registry import (
                register_aggregation,
                register_filter,
                register_metric,
            )

            @register_aggregation("issue1457-agg")
            def my_agg(items):
                return sum(items) / max(len(items), 1)

            @register_metric(
                metric="issue1457-metric",
                higher_is_better=True,
                aggregation="issue1457-agg",
            )
            def my_metric(items):
                return sum(1 for i in items if i)

            @register_filter("issue1457-filter")
            class MyFilter(Filter):
                def apply(self, resps, docs):
                    return resps
            """
        )
    )

    _include.import_user_modules(include_path=py_file)

    assert "issue1457-metric" in metric_registry
    assert "issue1457-filter" in filter_registry
    assert get_aggregation("issue1457-agg")([1, 1, 0, 1]) == 0.75


def test_cli_run_parses_include_module_multi(tmp_path: Path):
    import argparse

    from lm_eval._cli.run import Run

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    Run.create(subparsers)

    args = parser.parse_args(
        [
            "run",
            "--model",
            "dummy",
            "--tasks",
            "hellaswag",
            "--include_path",
            str(tmp_path / "x.py"),
            "--include_module",
            "pkg_a",
            "pkg_b.sub",
        ]
    )
    assert args.include_path == str(tmp_path / "x.py")
    assert args.include_module == ["pkg_a", "pkg_b.sub"]


def test_cli_ls_accepts_include_module():
    import argparse

    from lm_eval._cli.ls import List

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    List.create(subparsers)

    args = parser.parse_args(["ls", "tasks", "--include_module", "foo.bar"])
    assert args.include_module == ["foo.bar"]


def test_cli_validate_accepts_include_module():
    import argparse

    from lm_eval._cli.validate import Validate

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    Validate.create(subparsers)

    args = parser.parse_args(
        ["validate", "--tasks", "foo", "--include_module", "foo.bar"]
    )
    assert args.include_module == ["foo.bar"]
