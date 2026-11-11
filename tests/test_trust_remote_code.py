"""Tests for the trust_remote_code policy ("auto", True or False).

"auto" keeps the old behaviour, True opts in everywhere, and an explicit
False vetoes remote code that tasks ask for. Nothing here touches the network.
"""

import argparse
from types import SimpleNamespace

import datasets
import pytest

import lm_eval.utils as lm_utils
from lm_eval._cli.run import Run
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.evaluate_config import EvaluatorConfig


@pytest.fixture(autouse=True)
def reset_policy(monkeypatch):
    # The policy and the datasets flag are process-wide; restore both.
    monkeypatch.setattr(lm_utils, "_TRUST_REMOTE_CODE", "auto", raising=False)
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", "unset", raising=False
    )


def configure(trust_remote_code=None):
    kwargs = (
        {} if trust_remote_code is None else {"trust_remote_code": trust_remote_code}
    )
    return EvaluatorConfig(tasks=["xnli_va"], **kwargs)._configure()


def parse_cli(*argv):
    parser = argparse.ArgumentParser()
    Run.create(parser.add_subparsers())
    return EvaluatorConfig.from_cli(parser.parse_args(["run", "--tasks", "x", *argv]))


class TestConfig:
    def test_default_auto_is_unchanged(self):
        cfg = configure()
        assert cfg.trust_remote_code == "auto"
        assert "trust_remote_code" not in cfg.model_args
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE == "unset"

    def test_true_passes_through(self):
        cfg = configure(True)
        assert cfg.model_args["trust_remote_code"] is True
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is True

    def test_false_is_a_veto(self):
        cfg = configure(False)
        assert cfg.trust_remote_code is False
        assert "trust_remote_code" not in cfg.model_args
        assert datasets.config.HF_DATASETS_TRUST_REMOTE_CODE is False

    def test_model_args_untouched_by_auto(self):
        cfg = EvaluatorConfig(
            tasks=["x"], model_args={"trust_remote_code": True}
        )._configure()
        assert cfg.model_args["trust_remote_code"] is True

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="must be 'auto'"):
            configure("maybe")


class TestCLI:
    def test_flag_absent_means_auto(self):
        assert parse_cli().trust_remote_code == "auto"

    def test_bare_flag_means_true(self):
        assert parse_cli("--trust_remote_code").trust_remote_code is True

    @pytest.mark.parametrize("value, expected", [("false", False), ("True", True)])
    def test_explicit_value(self, value, expected):
        cfg = parse_cli("--trust_remote_code", value)
        assert cfg.trust_remote_code is expected


class TestTaskDownload:
    """`ConfigurableTask.download` with `dataset_kwargs: {trust_remote_code: true}`."""

    @pytest.fixture
    def task(self, monkeypatch):
        calls = []
        monkeypatch.setattr(datasets, "load_dataset", lambda **kw: calls.append(kw))
        return SimpleNamespace(
            config=SimpleNamespace(task="xnli_va", custom_dataset=None),
            DATASET_PATH="org/script-dataset",
            DATASET_NAME=None,
            calls=calls,
        )

    def download(self, task):
        ConfigurableTask.download(task, {"trust_remote_code": True})

    def test_false_raises_naming_task(self, task, monkeypatch):
        monkeypatch.setattr(datasets, "__version__", "3.6.0")
        configure(False)
        with pytest.raises(ValueError, match="Task 'xnli_va' needs remote code"):
            self.download(task)
        assert task.calls == []

    @pytest.mark.parametrize("policy", [None, True])
    def test_auto_and_true_load(self, task, monkeypatch, policy):
        monkeypatch.setattr(datasets, "__version__", "3.6.0")
        configure(policy)
        self.download(task)
        assert task.calls[0]["trust_remote_code"] is True

    def test_false_is_moot_on_datasets_4(self, task, monkeypatch):
        # datasets>=4 runs no dataset scripts, so the kwarg is dropped.
        monkeypatch.setattr(datasets, "__version__", "4.0.0")
        configure(False)
        self.download(task)
        assert "trust_remote_code" not in task.calls[0]


class FakeTokenizer:
    def __call__(self, text):
        return SimpleNamespace(input_ids=text.split())


@pytest.mark.parametrize(
    "module",
    ["lm_eval.tasks.ruler.common_utils", "lm_eval.tasks.babilong.common_utils"],
)
@pytest.mark.parametrize(
    "policy, expected", [(None, True), (True, True), (False, False)]
)
def test_tokenizer_helpers_read_policy(monkeypatch, module, policy, expected):
    mod = pytest.importorskip(module)
    seen = {}

    def from_pretrained(name, **kwargs):
        seen.update(kwargs)
        return FakeTokenizer()

    fake = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setattr(mod, "AutoTokenizer", fake)
    mod.get_tokenizer.cache_clear()
    try:
        configure(policy)
        mod.get_tokenizer("org/tokenizer")
    finally:
        mod.get_tokenizer.cache_clear()
    assert seen["trust_remote_code"] is expected
