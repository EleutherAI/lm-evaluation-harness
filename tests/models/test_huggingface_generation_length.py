"""Offline regression tests for checkpoint generation-length overrides."""

# Tokenizer special-token strings are vocabulary entries, not credentials.
# ruff: noqa: S106

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


@pytest.fixture(params=["causal", "seq2seq"])
def hf_model(request):
    """Use real local random models; no Hub access or mocked generation."""
    torch.manual_seed(0)
    if request.param == "causal":
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=8,
                n_positions=32,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=None,
                pad_token_id=0,
            )
        )
    else:
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=8,
                d_model=8,
                d_ff=16,
                d_kv=8,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=1,
                decoder_start_token_id=0,
                eos_token_id=None,
                pad_token_id=0,
            )
        )
    # Avoid random early stopping, so output length measures the actual budget.
    model.generation_config.suppress_tokens = [6]
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "a": 2,
                "b": 3,
                "c": 4,
                "x": 5,
                "<eos>": 6,
                "<unk>": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=32,
    )
    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend=request.param,
        device="cpu",
        max_length=32,
        batch_size=1,
        add_bos_token=False,
    )


@pytest.mark.parametrize("checkpoint_budget", [None, 8])
def test_model_generate_respects_max_length(hf_model, checkpoint_budget):
    hf_model.model.generation_config.max_new_tokens = checkpoint_budget
    original_config = hf_model.model.generation_config.to_dict()
    context = torch.tensor([[2, 3, 4]])

    output = hf_model._model_generate(context, max_length=5, stop=[])

    assert output.shape == (1, 5)
    assert hf_model.model.generation_config.to_dict() == original_config


def test_model_generate_preserves_explicit_max_new_tokens(hf_model):
    hf_model.model.generation_config.max_new_tokens = 8
    original_config = hf_model.model.generation_config.to_dict()
    context = torch.tensor([[2, 3, 4]])

    output = hf_model._model_generate(context, max_length=6, stop=[], max_new_tokens=2)

    expected_length = 5 if hf_model.backend == "causal" else 3
    assert output.shape == (1, expected_length)
    assert hf_model.model.generation_config.to_dict() == original_config


@pytest.mark.parametrize(
    "generation_kwargs,expected_length",
    [
        ({"max_gen_toks": 2}, 5),
        ({"max_new_tokens": 2}, 5),
        ({"max_gen_toks": 2, "max_length": 4}, 4),
    ],
)
def test_generate_until_overrides_checkpoint_budget(
    hf_model, generation_kwargs, expected_length, monkeypatch
):
    hf_model.model.generation_config.max_new_tokens = 8
    original_config = hf_model.model.generation_config.to_dict()
    outputs = []
    generate = hf_model.model.generate

    def record_generate(*args, **kwargs):
        output = generate(*args, **kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(hf_model.model, "generate", record_generate)
    request = Instance(
        request_type="generate_until",
        doc={},
        arguments=("a b c", generation_kwargs),
        idx=0,
    )

    hf_model.generate_until([request], disable_tqdm=True)

    # Preserve the existing total-length policy for both backends. In particular,
    # this regression does not redefine seq2seq decoder-length accounting.
    assert outputs[0].shape == (1, expected_length)
    assert hf_model.model.generation_config.to_dict() == original_config
