import copy
import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import BatchEncoding

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models import huggingface


@pytest.fixture
def audio_model(monkeypatch):
    # Isolate the audio backend's unrelated lazy-registry class-name mismatch.
    monkeypatch.setattr(
        "lm_eval.api.registry.register_model", lambda *names: lambda cls: cls
    )
    spec = importlib.util.spec_from_file_location(
        "audio_stop_test_backend",
        huggingface.__file__.replace("huggingface.py", "hf_audiolm.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.HFAUDIOLMQWEN.__new__(module.HFAUDIOLMQWEN)
    TemplateLM.__init__(model)
    model.tokenizer = SimpleNamespace(eos_token_id=2)
    model._max_length = 4096
    model.batch_size_per_gpu = 2
    model.truncation = False
    model.tok_encode = lambda text: list(range(len(text)))
    model.cache_hook = Mock()
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)
    return model


def run_generation(model, requests, decoded):
    contexts_seen = []

    def encode(contexts, audios, **kwargs):
        contexts_seen[:] = contexts
        ids = Mock(shape=(len(contexts), 3))
        ids.to.return_value = ids
        return BatchEncoding({"input_ids": ids})

    def generate(inputs, **kwargs):
        return SimpleNamespace(
            tolist=lambda: [[7, 8, 9, context] for context in contexts_seen]
        )

    model.tok_batch_multimodal_encode = encode
    model._model_multimodal_generate = generate
    model.tok_decode = lambda tokens, **kwargs: (
        "</s>" if isinstance(tokens, int) else decoded[tokens[0]]
    )
    return model.generate_until(requests, disable_tqdm=True)


@pytest.mark.parametrize(
    "until, decoded, expected",
    [
        (["\nQuestion:"], "42\nQuestion: next", "42"),
        ("STOP", "answerSTOPtrailing", "answer"),
        (["LATER", "EARLIER"], "answerEARLIERmiddleLATERtail", "answer"),
        (["STOP"], "STOPtail", ""),
        (["STOP"], "unchanged", "unchanged"),
        (["", "STOP"], "answerSTOPtail", "answer"),
        ([], "answer</s>tail", "answer"),
        (None, "answer</s>tail", "answer"),
    ],
)
def test_audio_generation_trims_stops_before_return_and_cache(
    audio_model, until, decoded, expected
):
    gen_kwargs = {} if until is None else {"until": until}
    original = copy.deepcopy(gen_kwargs)
    request = Instance(
        "generate_until", {}, ("context", gen_kwargs, {"audio": [{"array": [0.0]}]}), 0
    )

    assert run_generation(audio_model, [request], {"context": decoded}) == [expected]
    audio_model.cache_hook.add_partial.assert_called_once_with(
        "generate_until", ("context", gen_kwargs), expected
    )
    assert gen_kwargs == original


def test_audio_stops_preserve_request_order_across_batches(audio_model):
    contexts = ["a", "long context", "mid"]
    kwargs = [{"until": ["STOP"]}, {"until": ["STOP"]}, {"until": ["END"]}]
    requests = [
        Instance("generate_until", {}, (context, options, {"audio": []}), i)
        for i, (context, options) in enumerate(zip(contexts, kwargs, strict=True))
    ]
    decoded = {
        "a": "firstSTOPbatch padding",
        "long context": "secondSTOP",
        "mid": "thirdENDtail",
    }

    assert run_generation(audio_model, requests, decoded) == [
        "first",
        "second",
        "third",
    ]
    cached = {
        call.args[1][0]: call.args[2]
        for call in audio_model.cache_hook.add_partial.call_args_list
    }
    assert cached == dict(zip(contexts, ["first", "second", "third"], strict=True))
