import copy
from types import SimpleNamespace

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM
from lm_eval.models.winml import WindowsML


@pytest.fixture(params=[ONNXRuntimeGenAILM, WindowsML])
def genai_model(request):
    model = request.param.__new__(request.param)
    LM.__init__(model)
    model.model = object()
    model._max_gen_toks = 8
    model.max_length = 32
    model.tok_encode = lambda text: [ord(c) for c in text]
    model.tok_decode = lambda tokens: "".join(chr(t) for t in tokens)
    searches = []

    class Params:
        def __init__(self, model):
            self.options = {}

        def set_search_options(self, **options):
            self.options.update(options)
            searches.append(options)

    class Generator:
        def __init__(self, model, params):
            self.params = params
            self.tokens = []

        def append_tokens(self, tokens):
            self.tokens.extend(tokens)

        def is_done(self):
            return len(self.tokens) >= self.params.options["max_length"]

        def generate_next_token(self):
            self.tokens.append(ord("A"))

        def get_sequence(self, index):
            return self.tokens

    model.og = SimpleNamespace(GeneratorParams=Params, Generator=Generator)
    return model, searches


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, 8),
        ({"max_gen_toks": 3}, 3),
        ({"max_new_tokens": 3}, 3),
        ({"max_tokens": 3}, 3),
        ({"max_completion_tokens": 3}, 3),
        ({"max_new_tokens": "3"}, 3),
        ({"max_gen_toks": 2, "max_new_tokens": 3, "max_tokens": 4}, 2),
        ({"max_new_tokens": 3, "max_tokens": 4, "max_completion_tokens": 5}, 3),
        ({"max_tokens": 4, "max_completion_tokens": 5}, 4),
        ({"max_gen_toks": None, "max_new_tokens": 3}, 3),
    ],
)
def test_public_generation_honors_token_aliases(genai_model, kwargs, expected):
    model, searches = genai_model
    options = {"until": ["STOP"], **kwargs}
    original = copy.deepcopy(options)
    request = Instance("generate_until", {}, ("hi", options), 0)

    assert model.generate_until([request], disable_tqdm=True) == ["A" * expected]
    assert searches == [{"max_length": 2 + expected, "do_sample": False}]
    assert options == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"do_sample": False, "temperature": 0.7},
        {"do_sample": True, "temperature": 0.7},
    ],
)
def test_alias_preserves_context_cap_and_sampling(genai_model, kwargs):
    model, searches = genai_model
    model.max_length = 4
    request = Instance("generate_until", {}, ("hi", {"max_new_tokens": 3, **kwargs}), 0)

    assert model.generate_until([request], disable_tqdm=True) == ["AA"]
    expected = {"max_length": 4, "do_sample": kwargs.get("do_sample", False)}
    if expected["do_sample"]:
        expected.update(temperature=0.7, top_p=1.0, top_k=50)
    assert searches == [expected]
