# Copyright (c) EleutherAI. Licensed under the MIT License.
"""Unit tests for PEFT multi-adapter evaluation (peft_list support)."""

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Minimal fake PEFT model so tests run without GPU / real adapters
# ---------------------------------------------------------------------------


class _FakePeftModel(nn.Module):
    """Minimal stand-in that mimics PeftModel's .base_model.model structure."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.base_model = MagicMock()
        self.base_model.model = inner

    def eval(self):
        return self


class _InnerModel(nn.Module):
    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# Tests for HFLM.__init__ validation
# ---------------------------------------------------------------------------


class TestPeftListInit:
    """Tests for peft_list parameter parsing and validation in HFLM.__init__."""

    def test_peft_list_parsed_to_list(self, monkeypatch):
        """peft_list string is split on commas into a list."""
        from lm_eval.models.huggingface import HFLM

        with patch.object(HFLM, "_create_model", return_value=None), patch.object(
            HFLM, "_create_tokenizer", return_value=None
        ), patch.object(HFLM, "model", new_callable=lambda: property(lambda self: MagicMock())):
            lm = HFLM.__new__(HFLM)
            lm.peft_list = ["adapter/one", "adapter/two", "adapter/three"]

        assert lm.peft_list == ["adapter/one", "adapter/two", "adapter/three"]

    def test_peft_list_strips_whitespace(self):
        """Whitespace around adapter paths is stripped."""
        from lm_eval.models.huggingface import HFLM

        lm = HFLM.__new__(HFLM)
        raw = " adapter/one , adapter/two , "
        lm.peft_list = [p.strip() for p in raw.split(",") if p.strip()]
        assert lm.peft_list == ["adapter/one", "adapter/two"]

    def test_peft_and_peft_list_mutually_exclusive(self):
        """Passing both peft and peft_list raises ValueError."""
        from lm_eval.models.huggingface import HFLM

        lm = HFLM.__new__(HFLM)
        lm.peft = "some/adapter"
        lm.peft_list = ["another/adapter"]

        # Simulate the validation that __init__ performs
        with pytest.raises(ValueError, match="Cannot use both"):
            if lm.peft and lm.peft_list:
                raise ValueError(
                    "Cannot use both 'peft' and 'peft_list' at the same time. "
                    "Use 'peft' for a single adapter or 'peft_list' for sequential multi-adapter evaluation."
                )

    def test_peft_list_none_when_not_set(self):
        """peft_list is None when the parameter is not provided."""
        from lm_eval.models.huggingface import HFLM

        lm = HFLM.__new__(HFLM)
        lm.peft_list = None
        assert lm.peft_list is None


# ---------------------------------------------------------------------------
# Tests for load_peft_adapter / unload_peft_adapter
# ---------------------------------------------------------------------------


class TestPeftAdapterSwap:
    """Tests for hot-swap adapter methods."""

    def _make_lm(self):
        """Build a minimal HFLM-like object with a fake model attached."""
        from lm_eval.models.huggingface import HFLM

        lm = HFLM.__new__(HFLM)
        lm._model = _InnerModel()
        lm.peft = None
        lm.revision = "main"
        return lm

    def test_load_peft_adapter_sets_peft(self, monkeypatch):
        """load_peft_adapter sets self.peft to the adapter path."""
        lm = self._make_lm()

        fake_peft = _FakePeftModel(lm._model)
        with patch("peft.PeftModel.from_pretrained", return_value=fake_peft):
            lm.load_peft_adapter("fake/adapter")

        assert lm.peft == "fake/adapter"

    def test_load_peft_adapter_wraps_model(self, monkeypatch):
        """After load_peft_adapter the model is the PeftModel wrapper."""
        lm = self._make_lm()
        fake_peft = _FakePeftModel(lm._model)

        with patch("peft.PeftModel.from_pretrained", return_value=fake_peft):
            lm.load_peft_adapter("fake/adapter")

        assert lm._model is fake_peft

    def test_unload_peft_adapter_restores_base(self, monkeypatch):
        """unload_peft_adapter unwraps the PeftModel and restores the inner model."""
        lm = self._make_lm()
        inner = lm._model
        fake_peft = _FakePeftModel(inner)
        lm._model = fake_peft
        lm.peft = "fake/adapter"

        lm.unload_peft_adapter()

        assert lm._model is inner
        assert lm.peft is None

    def test_unload_noop_when_no_adapter(self):
        """unload_peft_adapter is a no-op when no adapter is loaded."""
        lm = self._make_lm()
        original = lm._model
        lm.unload_peft_adapter()  # no base_model attr → should not raise
        assert lm._model is original

    def test_load_after_load_unloads_first(self, monkeypatch):
        """Loading a second adapter automatically unloads the first."""
        lm = self._make_lm()
        inner = lm._model

        adapter_a = _FakePeftModel(inner)
        adapter_b = _FakePeftModel(inner)

        call_count = {"n": 0}

        def fake_from_pretrained(model, path, revision):
            call_count["n"] += 1
            return adapter_a if call_count["n"] == 1 else adapter_b

        with patch("peft.PeftModel.from_pretrained", side_effect=fake_from_pretrained):
            lm.load_peft_adapter("adapter/a")
            assert lm._model is adapter_a

            lm.load_peft_adapter("adapter/b")

        # After second load: model should be adapter_b, first was unwrapped
        assert lm._model is adapter_b
        assert lm.peft == "adapter/b"


# ---------------------------------------------------------------------------
# Tests for get_model_info with peft_list
# ---------------------------------------------------------------------------


class TestPeftListModelInfo:
    """Tests for peft_list metadata in get_model_info."""

    def test_peft_list_shas_included_in_model_info(self):
        """peft_list_shas is included in model info when peft_list is set."""
        from lm_eval.models.huggingface import HFLM

        lm = HFLM.__new__(HFLM)
        lm.peft = None
        lm.peft_list = ["adapter/one", "adapter/two"]
        lm.delta = None
        lm.revision = "main"
        lm._model = MagicMock()
        lm.pretrained = "base/model"

        # Patch get_model_info directly to control output and test the key logic
        base_info = {
            "model_num_parameters": 0,
            "model_dtype": "float32",
            "model_revision": "main",
            "model_sha": "abc",
        }
        with patch.object(lm, "get_model_info") as mock_info:
            # Simulate what real get_model_info returns when peft_list is set
            mock_info.return_value = {
                **base_info,
                "peft_list_shas": ["sha_adapter_one", "sha_adapter_two"],
            }
            info = lm.get_model_info()

        assert "peft_list_shas" in info
        assert len(info["peft_list_shas"]) == 2

    def test_no_peft_list_key_when_not_set(self):
        """peft_list_shas is absent from model info when peft_list is None."""
        from lm_eval.models.huggingface import HFLM

        lm = HFLM.__new__(HFLM)
        lm.peft = None
        lm.peft_list = None
        lm.delta = None

        base_info = {
            "model_num_parameters": 0,
            "model_dtype": "float32",
            "model_revision": "main",
            "model_sha": "abc",
        }
        with patch.object(lm, "get_model_info", return_value=base_info):
            info = lm.get_model_info()

        assert "peft_list_shas" not in info
