"""Exercise auto batch schedules through real local language-model scoring."""

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Create an offline causal model and tokenizer for public-API checks."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    backend = Tokenizer(
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
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **special_tokens)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=32,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    yield model, tokenizer
    torch.set_num_threads(previous_threads)


REQUESTS = [
    [("Hello", " world")],
    [("Hello", " world"), ("long", " prompt"), ("prompt", " world")],
    [
        ("Hello", continuation)
        for continuation in [
            " world",
            " prompt",
            " Hello",
            " long",
            " <eos>",
            " world prompt",
        ]
    ],
]


@pytest.mark.parametrize(
    "requests", REQUESTS, ids=["single", "three", "shared_context"]
)
@pytest.mark.parametrize("batch_size", ["auto", "auto:1", "auto:4", "auto:8", 2])
@pytest.mark.parametrize("logits_cache", [False, True])
def test_short_auto_batch_schedules(
    tiny_model_and_tokenizer, requests, batch_size, logits_cache
):
    """Adaptive batching must preserve every requested likelihood and greedy flag."""
    model, tokenizer = tiny_model_and_tokenizer
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_batch_size=2,
        max_length=32,
        add_bos_token=False,
        logits_cache=logits_cache,
        device="cpu",
    )
    expected = []
    for context, continuation in requests:
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
        inputs = torch.tensor([(context_ids + continuation_ids)[:-1]])
        with torch.no_grad():
            logits = model(inputs).logits[0, -len(continuation_ids) :]
        targets = torch.tensor(continuation_ids)
        logprob = -F.cross_entropy(logits, targets, reduction="sum")
        expected.append((logprob.item(), torch.equal(logits.argmax(-1), targets)))

    actual = lm.loglikelihood(
        [Instance("loglikelihood", {}, pair, 0) for pair in requests], disable_tqdm=True
    )

    assert len(actual) == len(expected)
    for (score, greedy), (reference_score, reference_greedy) in zip(
        actual, expected, strict=True
    ):
        assert score == pytest.approx(reference_score, abs=1e-6)
        assert greedy == reference_greedy
