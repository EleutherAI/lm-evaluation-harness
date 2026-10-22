"""Check that prompt batching preserves each prompt's BOS policy."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from lm_eval.models.huggingface import HFLM


def make_tokenizer(native_bos):
    """Create a real local tokenizer with an optional native BOS template."""
    special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "Hello": 4,
                "world": 5,
                "long": 6,
                "prompt": 7,
            },
            unk_token=special_tokens["unk_token"],
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A" if native_bos else "$A",
        special_tokens=[("<bos>", 1)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=4,
        **special_tokens,
    )


def make_lm(tokenizer, add_bos_token, backend="causal"):
    """Use the real tokenizer entry points without loading model weights."""
    lm = HFLM.__new__(HFLM)
    lm.tokenizer = tokenizer
    lm.backend = backend
    lm.add_bos_token = add_bos_token
    lm.custom_prefix_token_id = None
    return lm


@pytest.mark.parametrize("native_bos", [False, True])
@pytest.mark.parametrize("add_bos_token", [None, True, False])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_mixed_bos_batches(native_bos, add_bos_token, reverse, padding_side):
    """Adding or moving a neighboring prompt must not change valid tokens."""
    tokenizer = make_tokenizer(native_bos)
    lm = make_lm(tokenizer, add_bos_token)
    prompts = ["<bos>Hello", "Hello long world"]
    expected = [[1, 4], [4, 6, 5]]
    if native_bos and add_bos_token is not False:
        expected[1].insert(0, 1)
    if reverse:
        prompts.reverse()
        expected.reverse()

    input_ids, attention_mask = lm.tok_batch_encode(prompts, padding_side=padding_side)

    assert tokenizer.padding_side == "right"
    for row, mask, tokens in zip(input_ids, attention_mask, expected, strict=True):
        assert row[mask.bool()].tolist() == tokens
        assert mask.sum().item() == len(tokens)
        padding = row[~mask.bool()]
        assert torch.equal(padding, torch.zeros_like(padding))


@pytest.mark.parametrize("add_bos_token", [None, True, False])
@pytest.mark.parametrize("prefixed", [False, True])
def test_uniform_bos_batches(add_bos_token, prefixed):
    """Uniform batches retain their existing tokenization behavior."""
    tokenizer = make_tokenizer(True)
    lm = make_lm(tokenizer, add_bos_token)
    prefix = "<bos>" if prefixed else ""
    prompts = [prefix + "Hello", prefix + "Hello world"]
    input_ids, attention_mask = lm.tok_batch_encode(prompts)

    for prompt, row, mask in zip(prompts, input_ids, attention_mask, strict=True):
        assert row[mask.bool()].tolist() == lm.tok_encode(prompt)


@pytest.mark.parametrize("truncation", [False, True])
def test_mixed_bos_truncation(truncation):
    """Native truncation and explicit left truncation keep their token limits."""
    lm = make_lm(make_tokenizer(True), True)
    prompts = ["<bos>Hello long world prompt", "Hello long world prompt"]
    input_ids, attention_mask = lm.tok_batch_encode(
        prompts, truncation=truncation, left_truncate_len=3
    )
    expected = [4, 6, 5] if truncation else [6, 5, 7]
    assert input_ids.tolist() == [expected, expected]
    assert attention_mask.tolist() == [[1, 1, 1], [1, 1, 1]]


@pytest.mark.parametrize("add_bos_token", [None, True, False])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_seq2seq_bos_policy_unchanged(add_bos_token, padding_side):
    """Seq2seq inputs continue to use the tokenizer's native special tokens."""
    tokenizer = make_tokenizer(True)
    lm = make_lm(tokenizer, add_bos_token, backend="seq2seq")
    prompts = ["<bos>Hello", "Hello long world"]
    input_ids, attention_mask = lm.tok_batch_encode(prompts, padding_side=padding_side)
    for row, mask, expected in zip(
        input_ids, attention_mask, [[1, 1, 4], [1, 4, 6, 5]], strict=True
    ):
        assert row[mask.bool()].tolist() == expected
