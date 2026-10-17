"""Offline text-only adapter checks using a tiny, randomly initialized Mistral3."""

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lm_eval.api.instance import Instance
from lm_eval.models.mistral3 import Mistral3LM


@pytest.fixture
def tiny_mistral3():
    if not hasattr(transformers, "Mistral3ForConditionalGeneration"):
        pytest.skip("Mistral3 requires transformers >= 5.0.0")
    unk_token, bos_token, eos_token = "<unk>", "<bos>", "<eos>"
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                unk_token: 0,
                bos_token: 1,
                eos_token: 2,
                "alpha": 3,
                "beta": 4,
                "gamma": 5,
            },
            unk_token=unk_token,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=eos_token,
    )
    config = transformers.Mistral3Config(
        text_config=transformers.MistralConfig(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=2,
        ),
        vision_config=transformers.PixtralVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        ),
        image_token_index=99,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = transformers.Mistral3ForConditionalGeneration(config).eval()
    return model, tokenizer


@pytest.fixture(params=["preinitialized", "local_checkpoint"])
def adapter(request, tiny_mistral3, tmp_path):
    model, tokenizer = tiny_mistral3
    if request.param == "local_checkpoint":
        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        pretrained = str(tmp_path)
        tokenizer_arg = None
    else:
        pretrained = model
        tokenizer_arg = tokenizer
    return Mistral3LM(
        pretrained=pretrained,
        tokenizer=tokenizer_arg,
        device="cpu",
        dtype="float32",
        max_length=32,
        batch_size=2,
    )


def test_text_logits_are_causal_and_disable_gradients(adapter, tiny_mistral3):
    model, _ = tiny_mistral3
    tokens = torch.tensor([[3, 4, 5], [5, 4, 3]])
    with torch.enable_grad():
        expected = model(tokens).logits
        actual = adapter._model_call(tokens)
    assert adapter.backend == "causal"
    assert actual.shape == (2, 3, 6)
    assert expected.requires_grad
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected)


def test_batched_loglikelihood_matches_teacher_forcing(adapter, tiny_mistral3):
    model, tokenizer = tiny_mistral3
    pairs = [
        ("alpha beta", " gamma"),
        ("beta", " alpha gamma"),
        ("alpha beta", " alpha"),
    ]
    requests = [
        Instance(request_type="loglikelihood", doc={}, arguments=pair, idx=i)
        for i, pair in enumerate(pairs)
    ]
    actual = adapter.loglikelihood(requests, disable_tqdm=True)
    for (context, continuation), (score, greedy) in zip(pairs, actual, strict=True):
        context_ids = tokenizer.encode(context)
        continuation_ids = tokenizer.encode(continuation)
        tokens = torch.tensor([context_ids + continuation_ids])
        with torch.no_grad():
            logits = model(tokens[:, :-1]).logits[0, len(context_ids) - 1 :]
            logprobs = logits.log_softmax(dim=-1)
            expected = logprobs.gather(1, torch.tensor(continuation_ids)[:, None]).sum()
        assert score == pytest.approx(expected.item(), abs=1e-6)
        assert greedy == (logits.argmax(dim=-1).tolist() == continuation_ids)


def test_greedy_generation_returns_only_continuation(adapter, tiny_mistral3):
    model, tokenizer = tiny_mistral3
    context = "alpha beta"
    tokens = torch.tensor([tokenizer.encode(context)])
    with torch.no_grad():
        expected_ids = model.generate(
            tokens,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0, tokens.shape[1] :]
    request = Instance(
        request_type="generate_until",
        doc={},
        arguments=(context, {"max_gen_toks": 2, "until": []}),
        idx=0,
    )
    assert adapter.generate_until([request], disable_tqdm=True) == [
        tokenizer.decode(expected_ids, skip_special_tokens=True)
    ]
