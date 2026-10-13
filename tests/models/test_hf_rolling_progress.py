"""Offline rolling-progress regressions using a locally initialized HF model."""

import io
import math
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


INFERENCE_DESCRIPTION = "Running loglikelihood rolling requests"
TEXTS = ["a b a b a", "", "b c", "a b c a b c a b c"]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def lm():
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "[PAD]": 5},
            unk_token="[UNK]",  # noqa: S106 -- tokenizer vocabulary, not a secret
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",  # noqa: S106 -- tokenizer vocabulary, not a secret
        eos_token="[EOS]",  # noqa: S106 -- tokenizer vocabulary, not a secret
        pad_token="[PAD]",  # noqa: S106 -- tokenizer vocabulary, not a secret
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=6,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=1,
                eos_token_id=1,
                bos_token_id=1,
                pad_token_id=5,
            )
        )
    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=2,
        max_length=4,
        device="cpu",
    )


@pytest.fixture
def progress_bars(monkeypatch):
    bars = []

    class RecordingTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            self.requested_disable = kwargs.get("disable", False)
            self.requested_description = kwargs.get("desc", "")
            self.requested_unit = kwargs.get("unit", "it")
            self.output = io.StringIO()
            super().__init__(
                *args, **(kwargs | {"file": self.output, "mininterval": 0})
            )
            bars.append(self)

    monkeypatch.setattr("lm_eval.models.huggingface.tqdm", RecordingTqdm)
    return bars


def requests(texts):
    return [
        Instance("loglikelihood_rolling", {}, (text,), index)
        for index, text in enumerate(texts)
    ]


def reference_scores(lm, texts):
    scores = []
    for text in texts:
        windows = [
            (None, *utils.make_disjoint_window(window))
            for window in utils.get_rolling_token_windows(
                token_list=lm.tok_encode(text),
                prefix_token=lm.prefix_token_id,
                max_seq_len=lm.max_length,
                context_len=1,
            )
        ]
        scores.append(
            sum(
                score
                for score, _ in lm._loglikelihood_tokens(windows, disable_tqdm=True)
            )
        )
    return scores


@pytest.mark.parametrize("batch_size", [1, 2, 4, "auto"])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_rolling_progress_and_scores(
    lm, progress_bars, monkeypatch, batch_size, disabled, rank
):
    expected = reference_scores(lm, TEXTS)
    progress_bars.clear()
    lm.batch_size_per_gpu = batch_size
    lm._rank = rank
    lm.cache_hook = Mock()
    if batch_size == "auto":
        monkeypatch.setattr(lm, "_detect_batch_size", Mock(return_value=2))

    scores = lm.loglikelihood_rolling(requests(TEXTS), disable_tqdm=disabled)

    assert scores == pytest.approx(expected)
    assert scores[1] == 0
    assert [call.args for call in lm.cache_hook.add_partial.call_args_list] == [
        ("loglikelihood_rolling", (text,), score)
        for text, score in zip(TEXTS, scores, strict=True)
    ]
    (inference_bar,) = [
        bar
        for bar in progress_bars
        if bar.requested_description == INFERENCE_DESCRIPTION
    ]
    effective_batch_size = 2 if batch_size == "auto" else batch_size
    assert inference_bar.total == math.ceil(6 / effective_batch_size)
    assert inference_bar.requested_unit == "batch"
    assert inference_bar.requested_disable == (disabled or rank != 0)
    if not disabled and rank == 0:
        assert inference_bar.n == inference_bar.total
        assert inference_bar.output.getvalue()
    else:
        assert all(not bar.output.getvalue() for bar in progress_bars)
    inner_bars = [
        bar
        for bar in progress_bars
        if bar.requested_description == "Running loglikelihood requests"
    ]
    assert len(inner_bars) == inference_bar.total
    assert all(bar.requested_disable for bar in inner_bars)
    assert all(not bar.output.getvalue() for bar in inner_bars)
    if batch_size == "auto":
        lm._detect_batch_size.assert_called_once_with()


@pytest.mark.parametrize("texts", [[], [""], ["", ""]])
def test_empty_rolling_input_has_no_inference_batches(lm, progress_bars, texts):
    assert lm.loglikelihood_rolling(requests(texts), disable_tqdm=True) == [0] * len(
        texts
    )
    (inference_bar,) = [
        bar
        for bar in progress_bars
        if bar.requested_description == INFERENCE_DESCRIPTION
    ]
    assert inference_bar.total == 0
    assert all(not bar.output.getvalue() for bar in progress_bars)
    assert not any(
        bar.requested_description == "Running loglikelihood requests"
        for bar in progress_bars
    )


def test_rolling_progress_includes_distributed_padding(lm, progress_bars, monkeypatch):
    text = "a b a b a"
    (expected,) = reference_scores(lm, [text])
    progress_bars.clear()
    lm._world_size = 2
    monkeypatch.setattr(lm, "all_gather", lambda tensor: torch.tensor([2, 5]))

    (score,) = lm.loglikelihood_rolling(requests([text]))

    assert score == pytest.approx(expected)
    (inference_bar,) = [
        bar
        for bar in progress_bars
        if bar.requested_description == INFERENCE_DESCRIPTION
    ]
    assert inference_bar.total == inference_bar.n == 3
