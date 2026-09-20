import sys
from types import SimpleNamespace

import pytest

from lm_eval.loggers.wandb_logger import WandbLogger


def test_missing_wandb_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)

    with pytest.raises(ImportError, match=r"lm_eval\[wandb\]"):
        WandbLogger({"project": "test"})


def test_outdated_wandb_raises_minimum_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(__version__="0.16.2"))

    with pytest.raises(ImportError, match=r"wandb>=0\.16\.3"):
        WandbLogger({"project": "test"})
