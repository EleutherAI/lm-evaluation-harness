import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.api import registry
from lm_eval.api.instance import Instance


@pytest.fixture
def make_model(monkeypatch):
    """Load the real backend with the GPU engine and tokenizer replaced."""
    tokenizer = SimpleNamespace(pad_token_id=99, eos_token_id=99, decode=Mock())
    tokenizer.decode.side_effect = lambda ids, **kwargs: (
        "<eos>"
        if ids == 99
        else "".join(
            {
                1: "reasoning",
                2: " answer",
                3: "STOP",
                4: "tail",
                7: "</think>",
                0: "</think>",
            }[i]
            for i in ids
        )
    )
    modules = {
        "vllm": {
            "LLM": Mock(),
            "SamplingParams": SimpleNamespace,
            "TokensPrompt": Mock(),
        },
        "vllm.lora": {},
        "vllm.lora.request": {"LoRARequest": Mock()},
        "vllm.tokenizers": {"get_tokenizer": Mock(return_value=tokenizer)},
        "transformers": {"AutoConfig": Mock()},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.8.0" if name == "vllm" else version(name),
    )
    monkeypatch.setattr(registry, "register_model", lambda *names: lambda cls: cls)
    path = Path(__file__).parents[2] / "lm_eval/models/vllm_causallms.py"
    spec = importlib.util.spec_from_file_location("_vllm_thinking_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def create(think_end_token, token_ids, text, gen_kwargs=None):
        model = module.VLLM(
            pretrained="test-model",
            max_length=128,
            max_gen_toks=16,
            think_end_token=think_end_token,
        )
        model.tok_encode = Mock(return_value=[[10]])
        model._model_generate = Mock(
            return_value=[
                SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=token_ids, text=text)]
                )
            ]
        )
        model.cache_hook = Mock()
        request = Instance(
            request_type="generate_until",
            doc={},
            idx=0,
            arguments=("question", {"until": ["STOP"], **(gen_kwargs or {})}),
        )
        return model, request

    return create


@pytest.mark.parametrize(
    "marker,tokens,text,expected",
    [
        (7, [1, 7, 2], "reasoning</think> answer", "answer"),
        ("7", [1, 7, 2], "reasoning</think> answer", "answer"),
        (0, [1, 0, 2], "reasoning</think> answer", "answer"),
        ("0", [1, 0, 2], "reasoning</think> answer", "answer"),
        (7, [1, 7, 1, 7, 2], "reasoning</think>reasoning</think> answer", "answer"),
        (7, [1, 7], "reasoning</think>", ""),
        (7, [2], " answer", "answer"),
        (7, [2], "literal </think> text", "literal </think> text"),
        (7, [], "", ""),
        (7, [1, 3, 7, 2, 3, 4], "reasoningSTOP</think> answerSTOPtail", "answer"),
        (7, [1, 7, 2], "reasoning answer", "answer"),
        ("</think>", [1, 7, 2, 3, 4], "reasoning</think> answerSTOPtail", "answer"),
        (None, [2, 3, 4], " answerSTOPtail", " answer"),
        ("", [2, 3, 4], " answerSTOPtail", " answer"),
    ],
)
def test_thinking_token_output(make_model, marker, tokens, text, expected):
    model, request = make_model(marker, tokens, text)

    assert model.generate_until([request], disable_tqdm=True) == [expected]
    assert model.cache_hook.add_partial.call_args.args[2] == expected
    params = model._model_generate.call_args.kwargs["sampling_params"][0]
    assert params.stop == (
        ["STOP", "<eos>"] if marker is None or marker == "" else ["<eos>"]
    )


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_token_id_preserves_decode_option(make_model, skip_special_tokens):
    model, request = make_model(
        7, [1, 7, 2], "reasoning answer", {"skip_special_tokens": skip_special_tokens}
    )

    assert model.generate_until([request], disable_tqdm=True) == ["answer"]
    model.tokenizer.decode.assert_called_with(
        [2], skip_special_tokens=skip_special_tokens
    )


@pytest.mark.parametrize("spaces_between_special_tokens", [False, True])
def test_token_id_preserves_hf_spacing(make_model, spaces_between_special_tokens):
    model, request = make_model(
        7,
        [1, 7, 2],
        "reasoning answer",
        {"spaces_between_special_tokens": spaces_between_special_tokens},
    )
    model.tokenizer.name_or_path = "test-tokenizer"

    assert model.generate_until([request], disable_tqdm=True) == ["answer"]
    model.tokenizer.decode.assert_called_with(
        [2],
        skip_special_tokens=False,
        spaces_between_special_tokens=spaces_between_special_tokens,
        clean_up_tokenization_spaces=False,
    )
