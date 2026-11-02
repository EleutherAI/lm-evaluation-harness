"""Offline tokenizer contracts for the real TensorRT-LLM adapter."""

from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lm_eval.models import trtllm_causallms

UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {UNK: 0, BOS: 1, EOS: 2, "hello": 3, "world": 4, "test": 5},
            unk_token=UNK,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[(BOS, 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=UNK,
        bos_token=BOS,
        eos_token=EOS,
    )


@pytest.fixture
def make_adapter(monkeypatch, tokenizer):
    monkeypatch.setattr(trtllm_causallms, "find_spec", lambda _: object())
    monkeypatch.setattr(trtllm_causallms, "LLM", Mock(), raising=False)
    monkeypatch.setattr(trtllm_causallms, "KvCacheConfig", Mock(), raising=False)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))

    def make(**kwargs):
        return trtllm_causallms.TRTLLM(model="offline-model", **kwargs)

    return make


@pytest.mark.parametrize("tokenizer_adds_bos", [True, False])
def test_detects_tokenizer_bos_behavior(make_adapter, tokenizer, tokenizer_adds_bos):
    if not tokenizer_adds_bos:
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="$A"
        )

    adapter = make_adapter()

    assert adapter.add_bos_token is tokenizer_adds_bos
    assert adapter.tok_encode("hello world") == (
        [1, 3, 4] if tokenizer_adds_bos else [3, 4]
    )
    assert tokenizer.pad_token_id == tokenizer.unk_token_id


@pytest.mark.parametrize("add_bos", [True, False])
def test_explicit_bos_option_overrides_detection(make_adapter, add_bos):
    adapter = make_adapter(add_bos_token=add_bos)

    assert adapter.add_bos_token is add_bos
    assert adapter.tok_encode("hello") == ([1, 3] if add_bos else [3])


def test_mixed_bos_batch_preserves_order_without_duplicate_prefix(make_adapter):
    adapter = make_adapter()

    encoded = adapter.tok_encode(["world", "[BOS] hello", "hello world", "[BOS] world"])

    assert encoded == [[1, 4], [1, 3], [1, 3, 4], [1, 4]]
    assert adapter.tok_encode("[BOS] hello") == [1, 3]


@pytest.mark.parametrize("add_bos", [True, False])
@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_per_call_special_token_option_wins(make_adapter, add_bos, add_special_tokens):
    adapter = make_adapter(add_bos_token=add_bos)

    assert adapter.tok_encode("hello", add_special_tokens=add_special_tokens) == (
        [1, 3] if add_special_tokens else [3]
    )
    assert adapter.tok_encode("[BOS] hello", add_special_tokens=add_special_tokens) == [
        1,
        3,
    ]


@pytest.mark.parametrize("text", ["", []])
def test_empty_input_produces_no_tokens(make_adapter, text):
    assert make_adapter().tok_encode(text) == []
