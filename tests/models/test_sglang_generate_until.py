from lm_eval.api.instance import Instance
from lm_eval.models.sglang_causallms import SGLangLM


# kwargs every request carries once `modify_gen_kwargs` has normalized it
_BASE_KWARGS = {
    "temperature": 0.0,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
}


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


def _make_lm(outputs, encodings=None, **overrides):
    """Build an SGLangLM with the engine and tokenizer stubbed out."""
    lm = SGLangLM.__new__(SGLangLM)
    lm.batch_size = 8
    lm._rank = 0
    lm.add_bos_token = False
    lm._max_length = 32
    lm._max_gen_toks = 16
    lm.tokenizer = _FakeTokenizer()
    lm.think_end_token = None
    lm.truncation_side = "left"
    lm.cache_hook = _CacheHook()
    lm.seen_sampling_params = None
    lm.seen_requests = None
    for key, value in overrides.items():
        setattr(lm, key, value)

    def tok_encode(contexts, add_special_tokens=False):
        assert add_special_tokens is False
        if encodings is not None:
            return [list(enc) for enc in encodings]
        return [[idx + 1] for idx, _ in enumerate(contexts)]

    def model_generate(requests, generate=False, sampling_params=None):
        assert generate is True
        lm.seen_sampling_params = sampling_params
        lm.seen_requests = requests
        return [{"text": text} for text in outputs]

    lm.tok_encode = tok_encode
    lm._model_generate = model_generate
    return lm


def _requests(*gen_kwargs):
    return [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(f"ctx-{idx}", dict(kwargs)),
            idx=idx,
        )
        for idx, kwargs in enumerate(gen_kwargs)
    ]


def test_sglang_generate_until_uses_per_request_stop_sequences():
    lm = _make_lm(
        [
            "first. should be removed Question: not this stop",
            "second. keep this sentence Question: remove this",
        ]
    )
    requests = _requests(
        {"until": ["."], "max_gen_toks": 8},
        {"until": ["Question:"], "max_gen_toks": 8},
    )

    assert lm.generate_until(requests, disable_tqdm=True) == [
        "first",
        "second. keep this sentence ",
    ]
    assert lm.seen_sampling_params == [
        _BASE_KWARGS | {"max_tokens": 8, "stop": [".", "<eos>"]},
        _BASE_KWARGS | {"max_tokens": 8, "stop": ["Question:", "<eos>"]},
    ]
    assert lm.cache_hook.calls == [
        (
            "generate_until",
            ("ctx-0", _BASE_KWARGS | {"until": [".", "<eos>"], "max_gen_toks": 8}),
            "first",
        ),
        (
            "generate_until",
            (
                "ctx-1",
                _BASE_KWARGS | {"until": ["Question:", "<eos>"], "max_gen_toks": 8},
            ),
            "second. keep this sentence ",
        ),
    ]


def test_sglang_generate_until_withholds_task_stops_for_reasoning_models():
    """With `think_end_token` set, only EOS goes to the engine.

    Task-level stops such as the fewshot delimiter routinely occur inside
    `<think>` blocks; passing them to SGLang truncates the reasoning trace
    before any answer is produced. They are applied post-hoc instead.
    """
    lm = _make_lm(
        ["let me think\n\nstill thinking</think>Answer: 4\n\nQuestion: next"],
        think_end_token="</think>",
    )

    assert lm.generate_until(
        _requests({"until": ["\n\n"], "max_gen_toks": 8}), disable_tqdm=True
    ) == ["Answer: 4"]
    # engine sees EOS only ...
    assert lm.seen_sampling_params == [
        _BASE_KWARGS | {"max_tokens": 8, "stop": ["<eos>"]}
    ]
    # ... but the full stop list is still recorded for the cache key
    assert lm.cache_hook.calls[0][1][1]["until"] == ["\n\n", "<eos>"]


def test_sglang_generate_until_respects_truncation_side():
    for side, expected in (("left", [4, 5]), ("right", [1, 2])):
        lm = _make_lm(
            ["out"],
            encodings=[[1, 2, 3, 4, 5]],
            truncation_side=side,
            _max_length=10,
        )
        lm.generate_until(
            _requests({"until": ["Question:"], "max_gen_toks": 8}), disable_tqdm=True
        )
        assert lm.seen_requests == [expected], side


def test_sglang_generate_until_normalizes_gen_kwargs_aliases():
    lm = _make_lm(["out"])
    lm.generate_until(
        _requests({"until": "Question:", "max_new_tokens": 5, "do_sample": False}),
        disable_tqdm=True,
    )
    assert lm.seen_sampling_params == [
        _BASE_KWARGS | {"max_tokens": 5, "stop": ["Question:", "<eos>"]}
    ]
