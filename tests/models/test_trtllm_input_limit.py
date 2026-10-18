"""CPU coverage of TensorRT-LLM's separate input and sequence limits."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import transformers

from lm_eval.models import trtllm_causallms as trtllm


class CharacterTokenizer:
    """Use one token per character without downloads or special tokens."""

    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 1

    def decode(self, token):
        return "<bos>" if token == 1 else "<eos>"

    def __call__(self, strings, **kwargs):
        return SimpleNamespace(input_ids=[[ord(char) for char in s] for s in strings])


@pytest.fixture
def model(monkeypatch):
    """Run the real constructor with a local tokenizer and a recording engine."""
    engine = SimpleNamespace(
        generate=Mock(
            side_effect=lambda inputs, **kwargs: [
                SimpleNamespace(outputs=[SimpleNamespace(text="answer")])
                for _ in inputs
            ]
        )
    )
    factory = Mock(return_value=engine)
    monkeypatch.setattr(trtllm, "find_spec", lambda name: True)
    monkeypatch.setattr(trtllm, "LLM", factory, raising=False)
    monkeypatch.setattr(trtllm, "SamplingParams", SimpleNamespace, raising=False)
    monkeypatch.setattr(trtllm, "KvCacheConfig", SimpleNamespace, raising=False)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        Mock(return_value=CharacterTokenizer()),
    )
    model = trtllm.TRTLLM(
        model="unused",
        max_input_len=8,
        max_output_len=4,
        max_gen_toks=2,
        add_bos_token=False,
        enable_thinking=False,
    )
    assert factory.call_args.kwargs["max_input_len"] == 8
    assert factory.call_args.kwargs["max_seq_len"] == 12
    return model


@pytest.mark.parametrize(
    ("side", "expected"),
    [("left", "mnopqrst"), ("right", "abcdefgh"), ("middle", "abcdqrst")],
)
def test_long_prompt_respects_input_limit(model, side, expected):
    """Unused output capacity must not increase the permitted input length."""
    model.truncation_side = side
    request = SimpleNamespace(args=("abcdefghijklmnopqrst", {"until": ["\n"]}))
    assert model.generate_until([request], disable_tqdm=True) == ["answer"]
    call = model.llm.generate.call_args
    assert call.kwargs["inputs"] == [[ord(char) for char in expected]]
    assert call.kwargs["sampling_params"][0].max_tokens == 2


@pytest.mark.parametrize("prompt_length", [3, 8, 9, 20])
@pytest.mark.parametrize("generation_length", [1, 4, 6])
def test_input_and_sequence_limits(model, prompt_length, generation_length):
    """Keep short prompts and generation budgets while satisfying both caps."""
    request = SimpleNamespace(
        args=("a" * prompt_length, {"max_gen_toks": generation_length, "until": ["\n"]})
    )
    assert model.generate_until([request], disable_tqdm=True) == ["answer"]
    call = model.llm.generate.call_args
    tokens = call.kwargs["inputs"][0]
    params = call.kwargs["sampling_params"][0]
    assert len(tokens) <= 8
    assert len(tokens) + params.max_tokens <= 12
    assert len(tokens) == min(prompt_length, 8, 12 - generation_length)
    assert params.max_tokens == generation_length
