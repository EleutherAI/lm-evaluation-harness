import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from lm_eval.caching import cache


@pytest.fixture
def run_cache_subprocess(tmp_path):
    env = os.environ.copy()
    env.pop("LM_HARNESS_CACHE_PATH", None)
    env["XDG_CACHE_HOME"] = str(tmp_path / "user-cache")
    env["LOCALAPPDATA"] = str(tmp_path / "user-cache")
    env["PYTHONPATH"] = str(Path(cache.__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    def run(script, override=None):
        child_env = env.copy()
        if override is not None:
            child_env["LM_HARNESS_CACHE_PATH"] = override
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=tmp_path,
            env=child_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    return run


@pytest.mark.parametrize("override", [None, ""])
def test_default_cache_path_uses_user_cache_directory(run_cache_subprocess, override):
    run_cache_subprocess(
        """
        from pathlib import Path
        from platformdirs import user_cache_dir

        expected = Path(user_cache_dir("lm_eval", appauthor=False))
        existed_before_import = expected.exists()

        from lm_eval.caching import cache

        assert cache.PATH == str(expected)
        assert expected.exists() == existed_before_import
        assert cache.load_from_cache("not-cached") is None
        assert expected.exists() == existed_before_import
        """,
        override=override,
    )


@pytest.mark.parametrize(
    "override", ["relative-cache", "~/literal-cache", "  cache  ", "absolute"]
)
def test_cache_path_preserves_nonempty_override(
    run_cache_subprocess, tmp_path, override
):
    if override == "absolute":
        override = str(tmp_path / "custom-cache")
    run_cache_subprocess(
        """
        import os
        from pathlib import Path

        expected = os.environ["LM_HARNESS_CACHE_PATH"]
        existed_before_import = Path(expected).exists()

        from lm_eval.caching import cache

        assert cache.PATH == expected
        assert Path(expected).exists() == existed_before_import
        """,
        override=override,
    )


def test_cache_override_roundtrip(run_cache_subprocess, tmp_path):
    run_cache_subprocess(
        """
        from pathlib import Path
        from lm_eval.caching import cache

        cache_dir = Path(cache.PATH)
        assert not cache_dir.exists()
        assert cache.load_from_cache("roundtrip", cache=True) is None
        cache.delete_cache()
        assert not cache_dir.exists()

        cache.save_to_cache("roundtrip", {"ok": True})
        assert cache.load_from_cache("roundtrip") is None
        assert cache.load_from_cache("roundtrip", cache=True) == {"ok": True}
        assert len(list(cache_dir.iterdir())) == 1

        cache.delete_cache("roundtrip")
        assert cache.load_from_cache("roundtrip", cache=True) is None
        assert list(cache_dir.iterdir()) == []
        """,
        override=str(tmp_path / "custom-cache"),
    )


def test_invalid_explicit_cache_path_raises(run_cache_subprocess, tmp_path):
    invalid_path = tmp_path / "not-a-directory"
    invalid_path.write_text("keep me")
    run_cache_subprocess(
        """
        from pathlib import Path
        from lm_eval.caching import cache

        try:
            cache.save_to_cache("invalid", {"ok": True})
        except FileExistsError:
            pass
        else:
            raise AssertionError("expected invalid explicit cache path to fail")

        assert Path(cache.PATH).read_text() == "keep me"
        """,
        override=str(invalid_path),
    )


def test_save_to_cache_creates_missing_parent_directories(tmp_path, monkeypatch):
    nested_cache = tmp_path / "missing-parent" / "cache"
    monkeypatch.setattr(cache, "PATH", str(nested_cache))

    cache.save_to_cache("nested-key", {"ok": True})

    assert nested_cache.is_dir()
    assert cache.load_from_cache("nested-key", cache=True) == {"ok": True}


def test_long_cache_keys_are_hashed_below_filesystem_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    long_model_name = "tokenizer" + "very-long-model-name/" * 30
    cache_key = f"requests-mmlu-5shot-rank0-world_size1-chat_template-{long_model_name}"

    cache.save_to_cache(cache_key, {"ok": True})

    cache_files = os.listdir(tmp_path)
    assert len(cache_files) == 1
    assert len(cache_files[0].encode("utf-8")) <= cache.MAX_CACHE_FILENAME_BYTES
    assert cache.load_from_cache(cache_key, cache=True) == {"ok": True}


def test_short_cache_keys_keep_readable_name(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    cache_key = "requests-sciq-0shot-rank0-world_size1-tokenizergpt2"
    cache.save_to_cache(cache_key, ["cached"])

    cache_files = os.listdir(tmp_path)
    assert cache_files == [f"{cache_key}{cache.FILE_SUFFIX}"]
    assert cache.load_from_cache(cache_key, cache=True) == ["cached"]


def test_delete_cache_is_idempotent_when_directory_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path / "never-created"))

    cache.delete_cache()

    assert not (tmp_path / "never-created").exists()


def test_delete_cache_raises_when_path_is_a_regular_file(tmp_path, monkeypatch):
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("data")
    monkeypatch.setattr(cache, "PATH", str(file_path))

    try:
        cache.delete_cache()
    except NotADirectoryError:
        pass
    else:
        pytest.fail("expected NotADirectoryError when PATH is a regular file")


def test_delete_cache_removes_only_own_suffix_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    cache.save_to_cache("requests-task-key", {"ok": True})
    (tmp_path / "unrelated.txt").write_text("keep me")

    cache.delete_cache()

    assert (tmp_path / "unrelated.txt").exists()
    assert os.listdir(tmp_path) == ["unrelated.txt"]
