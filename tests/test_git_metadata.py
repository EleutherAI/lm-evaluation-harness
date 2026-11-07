"""Regression coverage for Git metadata when evaluating a pinned revision."""

import pytest

from lm_eval.loggers import utils


SHA1 = "a1" * 20
SHA256 = "b2" * 32


@pytest.fixture
def repository(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    return tmp_path


@pytest.mark.parametrize("commit", [SHA1, SHA256])
def test_read_detached_head_without_git(repository, commit):
    (repository / ".git" / "HEAD").write_text(commit + chr(10), encoding="utf-8")

    assert utils.get_commit_from_path(repository) == commit


def test_read_attached_head(repository):
    git_dir = repository / ".git"
    ref = git_dir / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text(SHA1 + chr(10), encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/main" + chr(10), encoding="utf-8")

    assert utils.get_commit_from_path(str(repository)) == SHA1


@pytest.mark.parametrize("head", ["", "not-a-commit", "a" * 39, "a" * 41])
def test_invalid_head_still_returns_none(repository, head):
    (repository / ".git" / "HEAD").write_text(head + chr(10), encoding="utf-8")

    assert utils.get_commit_from_path(repository) is None


def test_missing_repository_returns_none(tmp_path):
    assert utils.get_commit_from_path(tmp_path) is None


def test_git_command_fallback_preserves_detached_revision(repository, monkeypatch):
    (repository / ".git" / "HEAD").write_text(SHA1 + chr(10), encoding="utf-8")
    monkeypatch.chdir(repository)

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(utils.subprocess, "check_output", missing_git)

    assert utils.get_git_commit_hash() == SHA1


def test_environment_records_detached_parent_revision(repository, monkeypatch):
    (repository / ".git" / "HEAD").write_text(SHA1 + chr(10), encoding="utf-8")
    evaluation_dir = repository / "evaluation"
    evaluation_dir.mkdir()
    monkeypatch.chdir(evaluation_dir)
    monkeypatch.setattr("lm_eval.utils.is_torch_available", lambda: False)
    monkeypatch.setattr("lm_eval.utils.is_transformers_available", lambda: False)
    storage = {}

    utils.add_env_info(storage)

    assert storage["upper_git_hash"] == SHA1
