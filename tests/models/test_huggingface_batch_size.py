"""Regression tests for generation batch sizing without model downloads."""

from __future__ import annotations

from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest
import torch

from lm_eval.models.huggingface import HFLM


def make_lm():
    def encode(contexts, left_truncate_len=None, truncation=False):
        # Include a BOS token that the sorting-only tokenizer does not return.
        tokens = [[1] + [int(word) + 2 for word in text.split()] for text in contexts]
        if left_truncate_len:
            tokens = [seq[-left_truncate_len:] for seq in tokens]
        width = max(map(len, tokens))
        ids = torch.tensor([[0] * (width - len(seq)) + seq for seq in tokens])
        return ids, ids.ne(0).long()

    lm = SimpleNamespace(
        backend="causal",
        world_size=1,
        model=SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            generation_config=SimpleNamespace(),
        ),
        tokenizer=SimpleNamespace(bos_token=None),
        max_length=64,
        max_gen_toks=8,
        max_batch_size=16,
        truncation=False,
        tok_batch_encode=encode,
        tok_encode=lambda text: text.split(),
        tok_decode=lambda tokens, **kwargs: str(tokens),
        device=torch.device("cpu"),
        softmax_dtype=torch.float32,
        eot_token_id=0,
        rank=0,
        think_end_token=None,
        batch_size="auto",
        cache_hook=SimpleNamespace(add_partial=lambda *args: None),
    )
    if hasattr(HFLM, "_get_generation_probe_length"):
        lm._get_generation_probe_length = MethodType(
            HFLM._get_generation_probe_length, lm
        )
    lm._detect_batch_size = MethodType(HFLM._detect_batch_size, lm)
    return lm


def request(prompt, **kwargs):
    return SimpleNamespace(args=(prompt, kwargs))


@pytest.mark.parametrize(
    ("requests", "expected"),
    [
        ([request("1 2 3")], 12),
        ([request("1 2 3", max_new_tokens=4)], 8),
        ([request("1 2 3", max_tokens=5)], 9),
        ([request("1 2 3", max_completion_tokens=6)], 10),
        ([request("1 2 3 4", max_gen_toks=2), request("1", max_gen_toks=16)], 18),
        ([request(" ".join(["1"] * 80), max_gen_toks=8)], 64),
    ],
)
def test_probe_reserves_full_generation_horizon(requests, expected):
    """Use actual BOS/truncation rules and each request's own generation budget."""
    lm = make_lm()
    original = deepcopy(requests)
    assert lm._get_generation_probe_length(requests) == expected
    assert [req.args for req in requests] == [req.args for req in original]


@pytest.mark.parametrize(
    ("target", "name", "value"),
    [
        ("lm", "backend", "seq2seq"),
        ("lm", "device", SimpleNamespace(type="hpu")),
        ("lm", "world_size", 2),
        ("config", "use_cache", False),
        ("generation_config", "max_new_tokens", 1024),
        ("generation_config", "guidance_scale", 2.0),
        ("generation_config", "num_beams", 4),
        ("generation_config", "num_return_sequences", 2),
        ("generation_config", "cache_implementation", "static"),
        ("generation_config", "penalty_alpha", 0.6),
        ("generation_config", "prompt_lookup_num_tokens", 4),
        ("generation_config", "output_scores", True),
    ],
)
def test_special_modes_keep_existing_detector(target, name, value):
    lm = make_lm()
    obj = lm if target == "lm" else getattr(lm.model, target)
    setattr(obj, name, value)
    assert lm._get_generation_probe_length([request("1 2")]) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"do_sample": True},
        {"temperature": 0.7},
        {"max_length": 32},
        {"num_beams": 2},
        {"assistant_model": "assistant"},
        {"max_gen_toks": 64},
    ],
)
def test_any_unsupported_request_preserves_old_probe(kwargs):
    lm = make_lm()
    assert (
        lm._get_generation_probe_length([request("1"), request("2", **kwargs)]) is None
    )


def test_tensor_parallel_falls_back_even_with_world_size_one(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    lm = make_lm()
    assert lm._get_generation_probe_length([request("1")]) is None


def test_auto_batch_grows_for_short_requests():
    """A memory-limited model can fit larger short generations than full contexts."""
    lm = make_lm()
    lm.max_length = 1024
    lengths = []

    def forward(tokens, **kwargs):
        lengths.append(tokens.shape[1])
        if tokens.numel() > 1024:
            raise RuntimeError("CUDA out of memory.")
        return torch.zeros((*tokens.shape, 2))

    lm._model_call = forward
    assert lm._detect_batch_size() == 1
    horizon = lm._get_generation_probe_length([request("1 2 3")])
    assert horizon == 12
    assert lm._detect_batch_size(max_length=horizon) == 16
    assert set(lengths) == {12, 1024}


def test_loglikelihood_request_sizing_is_unchanged():
    lm = make_lm()
    lengths = []

    def forward(tokens, **kwargs):
        lengths.append(tokens.shape[1])
        return torch.zeros((*tokens.shape, 2))

    lm._model_call = forward
    lm._detect_batch_size([(("prompt", "target"), [1, 2, 3], [4, 5])])
    assert set(lengths) == {4}


def test_auto_generation_preserves_grouping_order_and_fixed_batches():
    lm = make_lm()
    probes = []
    batches = []
    lm.max_batch_size = 2
    detect = lm._detect_batch_size

    def detect_batch(**kwargs):
        probes.append(kwargs)
        return detect(**kwargs)

    def forward(tokens, **kwargs):
        if tokens.numel() > 64:
            raise RuntimeError("CUDA out of memory.")
        return torch.zeros((*tokens.shape, 2))

    lm._model_call = forward
    lm._detect_batch_size = detect_batch

    def generate(context, **kwargs):
        batches.append(context.shape[0])
        return torch.cat([context, context[:, -1:] + 10], dim=1)

    lm._model_generate = generate
    requests = [
        request("1", max_gen_toks=3),
        request("2 3 4", max_gen_toks=4),
        request("5 6", max_gen_toks=3),
    ]
    original = deepcopy(requests)
    automatic = HFLM.generate_until(lm, requests, disable_tqdm=True)
    assert batches == [2, 1]
    assert probes == [{"max_length": 8}]
    lm.batch_size = 2
    fixed = HFLM.generate_until(lm, requests, disable_tqdm=True)
    assert automatic == fixed == ["[13]", "[16]", "[18]"]
    assert len(probes) == 1
    assert [req.args for req in requests] == [req.args for req in original]


def test_mixed_bos_prefixes_keep_existing_probe():
    lm = make_lm()
    lm.tokenizer = SimpleNamespace(bos_token="<s>")  # noqa: S106 - tokenizer marker

    def encode(contexts, **kwargs):
        add_bos = not contexts[0].startswith("<s>")
        widths = [1 + text.startswith("<s>") + add_bos for text in contexts]
        ids = torch.ones((len(contexts), max(widths)), dtype=torch.long)
        return ids, ids.clone()

    lm.tok_batch_encode = encode
    prompts = [" a", "<s>a"]
    assert [encode([text])[0].shape[1] for text in prompts] == [2, 2]
    assert encode(prompts)[0].shape[1] == 3
    assert lm._get_generation_probe_length([request(text) for text in prompts]) is None


@pytest.mark.parametrize("name", ["num_beams", "num_return_sequences"])
def test_unset_generation_defaults_allow_greedy_probe(name):
    # Transformers 5 resolves unset (None) beam/return counts to one.
    lm = make_lm()
    setattr(lm.model.generation_config, name, None)
    assert lm._get_generation_probe_length([request("1 2 3")]) == 12
