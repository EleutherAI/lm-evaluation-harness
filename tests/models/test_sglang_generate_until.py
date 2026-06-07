from lm_eval.api.instance import Instance
from lm_eval.models.sglang_causallms import SGLangLM


class _FakeTokenizer:
    eos_token_id = 0

    def decode(self, token_id):
        assert token_id == 0
        return "<eos>"


class _CacheHook:
    def __init__(self):
        self.calls = []

    def add_partial(self, attr, req, res):
        self.calls.append((attr, req, res))


def test_sglang_generate_until_uses_per_request_stop_sequences():
    lm = SGLangLM.__new__(SGLangLM)
    lm.batch_size = 8
    lm._rank = 0
    lm.add_bos_token = False
    lm._max_length = 32
    lm._max_gen_toks = 16
    lm.tokenizer = _FakeTokenizer()
    lm.think_end_token = None
    lm.cache_hook = _CacheHook()
    lm.seen_sampling_params = None

    def tok_encode(contexts, add_special_tokens=False):
        assert add_special_tokens is False
        return [[idx + 1] for idx, _ in enumerate(contexts)]

    def modify_gen_kwargs(kwargs):
        return kwargs

    def model_generate(requests, generate=False, sampling_params=None):
        assert generate is True
        lm.seen_sampling_params = sampling_params
        assert requests == [[1], [2]]
        return [
            {"text": "first. should be removed Question: not this stop"},
            {"text": "second. keep this sentence Question: remove this"},
        ]

    lm.tok_encode = tok_encode
    lm.modify_gen_kwargs = modify_gen_kwargs
    lm._model_generate = model_generate

    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=("ctx-a", {"until": ["."], "max_gen_toks": 8}),
            idx=0,
        ),
        Instance(
            request_type="generate_until",
            doc={},
            arguments=("ctx-b", {"until": ["Question:"], "max_gen_toks": 8}),
            idx=1,
        ),
    ]

    assert lm.generate_until(requests, disable_tqdm=True) == [
        "first",
        "second. keep this sentence ",
    ]
    assert lm.seen_sampling_params == [
        {"max_tokens": 8, "stop": [".", "<eos>"]},
        {"max_tokens": 8, "stop": ["Question:", "<eos>"]},
    ]
    assert lm.cache_hook.calls == [
        (
            "generate_until",
            ("ctx-a", {"until": [".", "<eos>"], "max_gen_toks": 8}),
            "first",
        ),
        (
            "generate_until",
            ("ctx-b", {"until": ["Question:", "<eos>"], "max_gen_toks": 8}),
            "second. keep this sentence ",
        ),
    ]
