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


# --------------------------------------------------------------------------- chat templates


class _ChatTok:
    name_or_path = "org/model"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, history, **kwargs):
        self.calls.append((history, kwargs))
        return "rendered"


def test_rbln_chat_template_passes_args():
    lm = RBLNLM.__new__(RBLNLM)
    lm.tokenizer = _ChatTok()
    lm.chat_template_args = {"enable_thinking": False}
    assert lm.tokenizer_name == "org__model"
    assert lm.apply_chat_template([{"role": "user", "content": "hi"}]) == "rendered"
    _, kwargs = lm.tokenizer.calls[0]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                      "continue_final_message": False, "enable_thinking": False}


def test_rbln_vlm_chat_template_turns_placeholders_into_image_entries():
    from lm_eval.models.optimum_rbln_vlm import RBLNVLM

    lm = RBLNVLM.__new__(RBLNVLM)
    lm.processor = _ChatTok()
    lm.chat_template_args = {}
    lm.interleave, lm.max_images, lm.chat_applied = True, 999, False
    history = [{"role": "user", "content": "<image> What is this? <image> And this?"}]
    assert lm.apply_chat_template(history) == "rendered"
    assert lm.chat_applied
    content = lm.processor.calls[0][0][0]["content"]
    assert [c["type"] for c in content] == ["image", "text", "image", "text"]


@pytest.mark.parametrize(
    "text, until, think, expected",
    [
        ("<think>\nlong\n\nreasoning</think>\n\nAnswer: 4\n\nQ:", ["\n\nQ:"], "</think>", "Answer: 4"),
        ("Answer: 4\n\nQ:", ["\n\nQ:"], None, "Answer: 4"),
    ],
)
def test_think_end_token_postprocess(text, until, think, expected):
    from lm_eval.models.utils import postprocess_generated_text

    assert postprocess_generated_text(text, until, think) == expected


@pytest.mark.parametrize(
    "rbln_config, expected",
    [
        # decoder-only and Qwen-VL style: top level (the visual entry is ignored)
        ({"max_seq_len": 262144, "visual": {"max_seq_lens": [16384]}}, 262144),
        # gemma3 / pixtral / llava style composite
        ({"vision_tower": {}, "language_model": {"max_seq_len": 131072}}, 131072),
        ({"vision_model": {}, "text_model": {"max_seq_len": 131072}}, 131072),
        ({"vlm": {"max_seq_len": 114688}}, 114688),
        # attribute-style config objects (the loaded model's rbln_config)
        (types.SimpleNamespace(max_seq_len=None, language_model=types.SimpleNamespace(max_seq_len=8192)), 8192),
        ({"vlm": {}}, None),
        (None, None),
    ],
)
def test_compiled_max_seq_len(rbln_config, expected):
    assert rbln.compiled_max_seq_len(rbln_config) == expected


@pytest.mark.parametrize(
    "explicit, compiled, config, expected",
    [
        (None, 262144, types.SimpleNamespace(text_config=types.SimpleNamespace(max_position_embeddings=262144)), 262144),
        # composite VLM config without a compiled value: read text_config, not 2048
        (None, None, types.SimpleNamespace(text_config=types.SimpleNamespace(max_position_embeddings=32768)), 32768),
        (None, None, types.SimpleNamespace(max_position_embeddings=4096), 4096),
        (None, None, types.SimpleNamespace(), 2048),
        (1024, 262144, types.SimpleNamespace(), 1024),
    ],
)
def test_max_length(explicit, compiled, config, expected):
    lm = _lm(config)
    lm._max_length = explicit
    lm._compiled_max_seq_len = compiled
    assert lm.max_length == expected
