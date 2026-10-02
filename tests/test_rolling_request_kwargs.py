"""Offline regressions: real request/scoring code with CPU-only scoring stubs."""

from types import SimpleNamespace

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CacheHook, CachingLM, hash_args
from lm_eval.api.task import ConfigurableTask, PerplexityTask
from lm_eval.config.task import TaskConfig
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
    reject_rolling_options,
    rolling_context_len,
)


def task_stub(output_type="loglikelihood_rolling", **kwargs):
    task = object.__new__(ConfigurableTask)
    task._config = TaskConfig(task="offline", output_type=output_type, **kwargs)
    task.OUTPUT_TYPE = output_type
    task.doc_to_target = lambda doc: doc["text"]
    return task


def request(text, options=None):
    args = (text,) if options is None else (text, options)
    return Instance("loglikelihood_rolling", {}, args, 0)


class CPUHFLM(HFLM):
    """Exercise HFLM rolling aggregation without a tokenizer or model download."""

    max_length = 4
    prefix_token_id = -1
    batch_size = 2

    def __init__(self):
        LM.__init__(self)
        self.backend = "causal"
        self.windows = []

    def tok_encode(self, text, **kwargs):
        return list(range(len(text)))

    def _loglikelihood_tokens(self, requests, **kwargs):
        self.windows.extend(requests)
        # Context-sensitive synthetic scores detect accidental reuse of a cache.
        return [(-float(len(ctx) * len(target)), False) for _, ctx, target in requests]


@pytest.mark.parametrize("canonical", [None, {}, {"temperature": "0.25"}])
def test_generation_alias_precedence(canonical):
    config = TaskConfig(
        request_kwargs=canonical,
        generation_kwargs={"temperature": 0.9, "until": ["OLD"]},
    )
    expected = (
        {"temperature": 0.9, "until": ["OLD"]}
        if canonical is None
        else {**canonical, "until": ["\n\n"]}
    )
    if "temperature" in expected:
        expected["temperature"] = float(expected["temperature"])
    assert config.request_kwargs == expected
    assert config.generation_kwargs is config.request_kwargs
    assert config.to_dict()["request_kwargs"] == expected


def test_generation_defaults_and_override_replacement():
    from lm_eval.defaults import default_gen_kwargs

    task = task_stub("generate_until")
    assert task.config.request_kwargs == default_gen_kwargs("\n\n")
    task.set_config("generation_kwargs", {"until": ["STOP"]})
    assert task.construct_requests({}, "prompt").args == ("prompt", {"until": ["STOP"]})
    task.set_config("request_kwargs", {"temperature": 0.7}, update=True)
    assert task.config.generation_kwargs["temperature"] == 0.7


@pytest.mark.parametrize(
    "output_type",
    [
        "loglikelihood",
        "multiple_choice",
        "loglikelihood_rolling",
    ],
)
@pytest.mark.parametrize("legacy", [{"temperature": 0}, {"until": ["STOP"]}, {}])
def test_scoring_ignores_legacy_generation_options(output_type, legacy, caplog):
    config = TaskConfig(output_type=output_type, generation_kwargs=legacy)
    assert config.request_kwargs is None
    assert "Ignoring generation_kwargs" in caplog.text
    assert "request_kwargs" not in config.to_dict()


@pytest.mark.parametrize(
    "output_type",
    [
        "loglikelihood",
        "multiple_choice",
        "loglikelihood_rolling",
    ],
)
def test_scoring_rejects_explicit_canonical_generation_options(output_type):
    with pytest.raises(ValueError, match="request_kwargs"):
        TaskConfig(output_type=output_type, request_kwargs={"temperature": 0})


def test_rolling_alias_is_deliberately_narrow(caplog):
    config = TaskConfig(
        output_type="loglikelihood_rolling",
        generation_kwargs={"temperature": 0, "context_len": 3},
    )
    assert config.request_kwargs is None
    assert "Ignoring generation_kwargs" in caplog.text
    with pytest.raises(ValueError, match="context_len"):
        TaskConfig(
            output_type="loglikelihood_rolling", generation_kwargs={"context_len": 0}
        )
    with pytest.raises(ValueError, match="Unknown rolling"):
        TaskConfig(
            output_type="loglikelihood_rolling",
            generation_kwargs={"context_len": 3},
            request_kwargs={"contex_len": 2},
        )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"generation_kwargs": {"until": ["NEW"]}}, {"until": ["NEW"]}),
        (
            {"request_kwargs": {"temperature": "0.25"}},
            {"temperature": 0.25, "until": ["\n\n"]},
        ),
        (
            {"generation_kwargs": {"until": ["NEW"]}, "request_kwargs": {}},
            {"until": ["\n\n"]},
        ),
        (
            {
                "generation_kwargs": {"until": ["NEW"]},
                "request_kwargs": {"until": ["CANONICAL"]},
            },
            {"until": ["CANONICAL"]},
        ),
    ],
)
def test_class_config_constructor_alias_layers_and_isolation(overrides, expected):
    class Documents(list):
        @property
        def features(self):
            return {"text": None}

    class OfflineConfiguredTask(ConfigurableTask):
        CONFIG = TaskConfig(
            task="offline",
            generation_kwargs={"until": ["OLD"]},
            metric_list=[],
        )

        def download(self, *args, **kwargs):
            pass

        def fewshot_docs(self):
            return None

        @property
        def eval_docs(self):
            return Documents([{"text": "answer"}])

        def doc_to_text(self, doc):
            return "prompt"

        def doc_to_target(self, doc):
            return doc["text"]

    task = OfflineConfiguredTask(config=overrides)
    second = OfflineConfiguredTask()
    req = task.construct_requests({"text": "answer"}, "prompt")
    assert req.args == ("prompt", expected)
    assert task.config.generation_kwargs is task.config.request_kwargs
    assert task.dump_config()["generation_kwargs"] == expected
    assert task.dump_config()["request_kwargs"] == expected
    req.args[1]["until"].append("REQUEST MUTATION")
    assert task.dump_config()["request_kwargs"] == expected
    task.config.request_kwargs["until"].append("INSTANCE MUTATION")
    assert second.construct_requests({}, "prompt").args == (
        "prompt",
        {"until": ["OLD"]},
    )
    assert second.dump_config()["request_kwargs"] == {"until": ["OLD"]}
    assert OfflineConfiguredTask.CONFIG.to_dict()["request_kwargs"] == {
        "until": ["OLD"]
    }
    assert all(
        "INSTANCE MUTATION" not in value.get("until", [])
        for value in overrides.values()
    )


class OfflinePerplexityTask(PerplexityTask):
    def download(self, *args, **kwargs):
        pass

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True


def test_perplexity_task_options_match_metadata_and_actual_histories():
    task = OfflinePerplexityTask()
    task._config = TaskConfig(
        task="offline",
        output_type="loglikelihood_rolling",
        request_kwargs={"context_len": 3},
    )
    req = task.construct_requests("abcdefghij", "")
    assert req.args == ("abcdefghij", {"context_len": 3})
    assert req.args[1] == task.dump_config()["request_kwargs"]
    model = CPUHFLM()
    assert model.loglikelihood_rolling([req], disable_tqdm=True) == [-22.0]
    assert model.windows == [
        (None, [-1], [0, 1, 2, 3]),
        (None, [1, 2, 3], [4, 5]),
        (None, [3, 4, 5], [6, 7]),
        (None, [5, 6, 7], [8, 9]),
    ]
    req.args[1]["context_len"] = 2
    assert task.dump_config()["request_kwargs"] == {"context_len": 3}
    task.set_config("request_kwargs", {"typo": 3})
    with pytest.raises(ValueError, match="Unknown rolling"):
        task.construct_requests("abc", "")


@pytest.mark.parametrize("options", [None, {}, {"context_len": 1}])
def test_perplexity_task_explicit_defaults_keep_legacy_shape(options):
    task = OfflinePerplexityTask()
    task._config = TaskConfig(
        output_type="loglikelihood_rolling", request_kwargs=options
    )
    assert task.construct_requests("abc", "").args == ("abc",)


def test_legacy_perplexity_defaults_do_not_fingerprint_generation_kwargs(monkeypatch):
    import lm_eval.api.task as task_module

    keys = []

    def load_from_cache(file_name, cache):
        keys.append(file_name)
        return [[request("cached")]]

    monkeypatch.setattr(task_module, "load_from_cache", load_from_cache)
    task = OfflinePerplexityTask()
    assert task.config.output_type == "generate_until"
    assert task.construct_requests("abcdefghij", "").args == ("abcdefghij",)
    model = CPUHFLM()
    assert model.loglikelihood_rolling([task.construct_requests("abcdefghij", "")]) == [
        -14.0
    ]
    task.build_all_requests(cache_requests=True)
    task.config.request_kwargs["temperature"] = 0.5
    task.build_all_requests(cache_requests=True)
    assert keys[0] == keys[1]
    assert "-rolling" not in keys[0]
    task._config = TaskConfig(output_type="loglikelihood_rolling")
    task.build_all_requests(cache_requests=True)
    assert keys[2] == keys[0]
    task.set_config("request_kwargs", {"context_len": 3})
    task.build_all_requests(cache_requests=True)
    assert keys[3] != keys[0]


@pytest.mark.parametrize("cli", [{"temperature": 0.4}, "temperature=0.4"])
def test_public_generation_cli_precedence(monkeypatch, cli):
    from lm_eval import evaluator

    generation = task_stub(
        "generate_until", request_kwargs={"temperature": 0.1, "until": ["END"]}
    )
    rolling = task_stub(request_kwargs={"context_len": 3})
    loaded = {"tasks": {"generation": generation, "rolling": rolling}, "groups": {}}
    monkeypatch.setattr(evaluator, "_log_selected_tasks", lambda *args: None)

    class ReachedEvaluation(Exception):
        pass

    def check_overrides(**kwargs):
        assert generation.construct_requests({}, "prompt").args[1] == {
            "temperature": 0.4,
            "until": ["END"],
        }
        assert rolling.config.request_kwargs == {"context_len": 3}
        raise ReachedEvaluation

    monkeypatch.setattr(evaluator, "evaluate", check_overrides)
    with pytest.raises(ReachedEvaluation):
        evaluator.simple_evaluate(
            model=CPUHFLM(),
            tasks=["offline"],
            gen_kwargs=cli,
            task_manager=SimpleNamespace(load=lambda tasks: loaded),
            random_seed=None,
            numpy_random_seed=None,
            torch_random_seed=None,
            fewshot_random_seed=None,
        )


@pytest.mark.parametrize("options", [None, {}, {"context_len": 1}])
def test_legacy_request_shape(options):
    task = task_stub(request_kwargs=options)
    assert task.construct_requests({"text": "abc"}, "").args == ("abc",)


def test_rolling_alias_and_request_copy():
    task = task_stub(generation_kwargs={"context_len": 3})
    req = task.construct_requests({"text": "abc"}, "")
    assert req.args == ("abc", {"context_len": 3})
    req.args[1]["context_len"] = 2
    assert task.dump_config()["request_kwargs"] == {"context_len": 3}
    assert (
        task_stub(
            request_kwargs={}, generation_kwargs={"context_len": 3}
        ).config.request_kwargs
        == {}
    )


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", None])
def test_invalid_context(value):
    error = (
        TypeError
        if isinstance(value, bool) or not isinstance(value, int)
        else ValueError
    )
    with pytest.raises(error, match="context_len"):
        TaskConfig(
            output_type="loglikelihood_rolling", request_kwargs={"context_len": value}
        )


@pytest.mark.parametrize(
    "options", [{"stride": 2}, {"stride": 2, "context_len": 3}, {"typo": 1}]
)
def test_unknown_options(options):
    with pytest.raises(ValueError, match="Unknown rolling"):
        rolling_context_len(options, 4)


@pytest.mark.parametrize("options", [[], "context_len=2", 3])
def test_non_mapping_options(options):
    with pytest.raises(TypeError):
        rolling_context_len(options, 4)


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 7, 8, 9, 13])
@pytest.mark.parametrize("context", [1, 2, 3, 4])
def test_actual_hf_windows_are_causal_and_score_tokens_once(length, context):
    model = CPUHFLM()
    text = "x" * length
    model.loglikelihood_rolling(
        [request(text, {"context_len": context})], disable_tqdm=True
    )
    targets = [token for _, _, target in model.windows for token in target]
    assert targets == list(range(length))
    for i, (_, ctx, target) in enumerate(model.windows):
        assert ctx
        assert len(ctx) + len(target) <= model.max_length + 1
        if i == 0:
            assert ctx == [-1]
            assert target == list(range(min(length, 4)))
        else:
            assert ctx == list(range(target[0] - len(ctx), target[0]))
            assert len(ctx) >= context
            assert len(ctx) + len(target) == 5
        # Each scored target sees only the context and earlier targets.
        for j, token in enumerate(target):
            assert all(previous < token for previous in ctx + target[:j])


def test_exact_default_and_tail_windows():
    model = CPUHFLM()
    model.loglikelihood_rolling([request("x" * 10)], disable_tqdm=True)
    assert model.windows == [
        (None, [-1], [0, 1, 2, 3]),
        (None, [3], [4, 5, 6, 7]),
        (None, [5, 6, 7], [8, 9]),
    ]
    expected = [
        (None,) + make_disjoint_window(pair)
        for pair in get_rolling_token_windows(list(range(10)), -1, 4, 1)
    ]
    assert model.windows == expected


def test_hf_validates_before_scoring():
    model = CPUHFLM()
    with pytest.raises(ValueError, match="context_len"):
        model.loglikelihood_rolling([request("abc"), request("", {"context_len": 5})])
    assert model.windows == []


def test_mixed_requests_and_original_partial_cache_keys():
    model = CPUHFLM()
    db = {}
    model.cache_hook = CacheHook(SimpleNamespace(dbdict=db))
    requests = [
        request("abcdefghij"),
        request(""),
        request("abcdefghij", {"context_len": 3}),
        request("x"),
    ]
    results = model.loglikelihood_rolling(requests, disable_tqdm=True)
    assert results == [-14.0, 0, -22.0, -1.0]
    assert len(db) == 4
    for req, result in zip(requests, results, strict=True):
        assert db[hash_args("loglikelihood_rolling", req.args)] == result
    assert requests[2].args[1] == {"context_len": 3}


def test_response_cache_roundtrip(tmp_path):
    model = CPUHFLM()
    cached = CachingLM(model, str(tmp_path / "rolling.db"))
    requests = [request("abcdefghij"), request("abcdefghij", {"context_len": 3})]
    try:
        assert cached.loglikelihood_rolling(requests) == [-14.0, -22.0]
        model.windows.clear()
        assert cached.loglikelihood_rolling(requests[::-1]) == [-22.0, -14.0]
        assert model.windows == []
    finally:
        cached.dbdict.close()


def test_request_construction_cache_separates_options(monkeypatch):
    import lm_eval.api.task as task_module

    keys = []
    cached = [[request("cached")]]

    def load_from_cache(file_name, cache):
        keys.append(file_name)
        return cached

    monkeypatch.setattr(task_module, "load_from_cache", load_from_cache)
    for options in [None, {}, {"context_len": 2}, {"context_len": 3}]:
        task_stub(request_kwargs=options).build_all_requests(cache_requests=True)
    assert keys[0] == keys[1]
    assert len(set(keys)) == 3


def test_unsupported_helper_and_seq2seq():
    reject_rolling_options([request("abc")], "legacy")
    with pytest.raises(ValueError, match="legacy.*HFLM"):
        reject_rolling_options(
            [request("abc"), request("abc", {"context_len": 2})], "legacy"
        )
    model = CPUHFLM()
    model.backend = "seq2seq"
    with pytest.raises(ValueError, match="seq2seq.*HFLM"):
        model.loglikelihood_rolling([request("abc", {"context_len": 2})])
    assert model.windows == []


def test_template_api_rejects_before_tokenization():
    from lm_eval.models.api_models import TemplateAPI

    with pytest.raises(ValueError, match="TemplateAPI.*HFLM"):
        TemplateAPI.loglikelihood_rolling(
            SimpleNamespace(), [request("abc", {"context_len": 2})]
        )


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("lm_eval.models.vllm_causallms", "VLLM"),
        ("lm_eval.models.sglang_causallms", "SGLangLM"),
    ],
)
def test_optional_backends_reject_before_initialization(module_name, class_name):
    module = pytest.importorskip(module_name)
    with pytest.raises(ValueError, match="does not support rolling request_kwargs"):
        getattr(module, class_name).loglikelihood_rolling(
            SimpleNamespace(), [request("abc", {"context_len": 2})]
        )


def test_evaluator_rejects_legacy_backend_before_dispatch(monkeypatch):
    from lm_eval import evaluator

    req = request("abc", {"context_len": 2})
    req.repeats = 1
    task = SimpleNamespace(
        instances=[req],
        build_all_requests=lambda **kwargs: None,
    )
    monkeypatch.setattr(evaluator, "get_sample_size", lambda *args: None)
    # No scoring method: the capability guard must fail before dispatch.
    model = SimpleNamespace(rank=0, world_size=1)
    with pytest.raises(ValueError, match="does not support rolling request_kwargs"):
        evaluator.evaluate(
            lm=model,
            task_dict={"tasks": {"offline": task}, "groups": {}},
            write_out=False,
        )


def test_explicit_default_direct_request_keeps_original_cache_key():
    model = CPUHFLM()
    db = {}
    model.cache_hook = CacheHook(SimpleNamespace(dbdict=db))
    legacy = request("abcdefghij")
    explicit = request("abcdefghij", {"context_len": 1})
    assert model.loglikelihood_rolling([legacy, explicit], disable_tqdm=True) == [
        -14.0,
        -14.0,
    ]
    assert hash_args("loglikelihood_rolling", legacy.args) in db
    assert hash_args("loglikelihood_rolling", explicit.args) in db
    assert len(db) == 2


@pytest.mark.parametrize("context", [1, 3, 4])
@pytest.mark.parametrize("length", [0, 1, 4, 5, 9])
def test_tiny_cpu_model_matches_independent_per_token_reference(context, length):
    """Score locally initialized weights; no pretrained model or dataset downloads."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(1565)
    unknown_token, end_token = "[UNK]", "[EOS]"
    vocabulary = {unknown_token: 0, end_token: 1, **{f"t{i}": i + 2 for i in range(12)}}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token=unknown_token))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token=unknown_token,
        eos_token=end_token,
        pad_token=end_token,
    )
    network = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=8,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).eval()
    model = HFLM(
        pretrained=network,
        tokenizer=tokenizer,
        device="cpu",
        max_length=4,
        batch_size=2,
    )
    text = " ".join(f"t{i}" for i in range(length))
    tokens = model.tok_encode(text)
    assert tokens == list(range(2, length + 2))
    result = model.loglikelihood_rolling(
        [request(text, {"context_len": context})], disable_tqdm=True
    )[0]
    expected = 0.0
    capacity = 4
    stride = capacity - context + 1
    # Assign each target position independently, without calling the window helper.
    with torch.no_grad():
        for position, target in enumerate(tokens):
            if position < capacity:
                history = [model.prefix_token_id, *tokens[:position]]
            else:
                group = 1 + (position - capacity) // stride
                end = min(length, capacity + group * stride)
                history = tokens[end - capacity - 1 : position]
            assert 1 <= len(history) <= capacity
            logits = network(torch.tensor([history])).logits[0, -1]
            expected += torch.log_softmax(logits, dim=-1)[target].item()
    assert result == pytest.approx(expected, abs=5e-6, rel=0)
    if context == 1:
        assert (
            model.loglikelihood_rolling([request(text)], disable_tqdm=True)[0] == result
        )
