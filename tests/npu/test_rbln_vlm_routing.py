"""CPU-only checks for the rbln / rbln-vlm loading and generation helpers.

No model is loaded: the helpers are called on bare instances with stub
configs, tokenizers and auto classes.
"""

import types

import pytest

import lm_eval.models.optimum_rbln as rbln
from lm_eval.models.optimum_rbln import RBLNLM


class _ImageTextToText:
    _model_mapping_names = {"qwen3_vl": "Qwen3VLForConditionalGeneration",
                            "gemma3": "Gemma3ForConditionalGeneration"}


class _Vision2Seq:
    _model_mapping_names = {"blip-2": "Blip2ForConditionalGeneration"}


class _NoMapping:
    pass


def _vlm(model_type):
    lm = RBLNLM.__new__(RBLNLM)
    lm.model_type = "vlm"
    lm._config = types.SimpleNamespace(model_type=model_type)
    return lm


def _patch(monkeypatch, itt, v2s):
    monkeypatch.setattr(rbln, "RBLNAutoModelForImageTextToText", itt)
    monkeypatch.setattr(rbln, "RBLNAutoModelForVision2Seq", v2s)


def test_vlm_routed_by_mapping(monkeypatch):
    _patch(monkeypatch, _ImageTextToText, _Vision2Seq)
    assert _vlm("qwen3_vl")._get_rbln_model_class("x") is _ImageTextToText
    assert _vlm("blip-2")._get_rbln_model_class("x") is _Vision2Seq


def test_vlm_without_vision2seq_uses_image_text_to_text(monkeypatch):
    # optimum-rbln 0.11 has no RBLNAutoModelForVision2Seq.
    _patch(monkeypatch, _ImageTextToText, None)
    assert _vlm("qwen3_vl")._get_rbln_model_class("x") is _ImageTextToText
    assert _vlm("qwen3_5")._get_rbln_model_class("x") is _ImageTextToText


def test_vlm_without_mappings_keeps_old_routing(monkeypatch):
    itt = type("ITT", (_NoMapping,), {})
    v2s = type("V2S", (_NoMapping,), {})
    _patch(monkeypatch, itt, v2s)
    assert _vlm("gemma3")._get_rbln_model_class("x") is itt
    assert _vlm("llava")._get_rbln_model_class("x") is v2s


def test_vlm_without_any_auto_class_raises(monkeypatch):
    _patch(monkeypatch, None, None)
    with pytest.raises(ImportError):
        _vlm("qwen3_vl")._get_rbln_model_class("x")


class _Tok:
    eos_token_id = 2

    def decode(self, ids):
        return f"<{ids[0]}>"


def _lm(config):
    lm = RBLNLM.__new__(RBLNLM)
    lm.tokenizer = _Tok()
    lm._config = config
    return lm


@pytest.mark.parametrize(
    "config, expected",
    [
        (types.SimpleNamespace(eos_token_id=7), ["<7>"]),
        (types.SimpleNamespace(eos_token_id=[7, 8]), ["<7>", "<8>"]),
        # transformers 5 composite VLM config: eos only under text_config
        (types.SimpleNamespace(text_config=types.SimpleNamespace(eos_token_id=9)), ["<9>"]),
        (types.SimpleNamespace(), ["<2>"]),
    ],
)
def test_eos_stop_strings(config, expected):
    lm = _lm(config)
    assert lm._eos_stop_strings() == expected


class _Gen:
    """Stub model: records generate kwargs; optionally rejects token_type_ids."""

    def __init__(self, rejects):
        self.rejects, self.calls = rejects, []

    def generate(self, **kwargs):
        self.calls.append(sorted(kwargs))
        if self.rejects and "token_type_ids" in kwargs:
            raise ValueError("The following `model_kwargs` are not used by the model: ['token_type_ids']")
        return "out"


def _vlm_gen(rejects):
    from lm_eval.models.optimum_rbln_vlm import RBLNVLM

    lm = RBLNVLM.__new__(RBLNVLM)
    lm._text_token_type_ids = None
    lm.model = _Gen(rejects)
    return lm


@pytest.mark.parametrize("rejects", [False, True])
def test_text_generate_token_type_ids(rejects):
    torch = pytest.importorskip("torch")
    lm = _vlm_gen(rejects)
    ids = torch.tensor([[1, 2]])
    assert lm._text_generate(input_ids=ids, max_new_tokens=1) == "out"
    assert lm._text_generate(input_ids=ids, max_new_tokens=1) == "out"
    assert lm._text_token_type_ids is (not rejects)
    # A rejecting model is tried with token_type_ids once, then never again.
    with_tti = [c for c in lm.model.calls if "token_type_ids" in c]
    assert len(with_tti) == (1 if rejects else 2)


def test_text_generate_other_errors_propagate():
    torch = pytest.importorskip("torch")
    lm = _vlm_gen(False)
    lm.model.generate = lambda **kw: (_ for _ in ()).throw(ValueError("something else"))
    with pytest.raises(ValueError, match="something else"):
        lm._text_generate(input_ids=torch.tensor([[1]]))
