from types import SimpleNamespace

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.model import LM, CachingLM, GenerationResult, hash_args
from lm_eval.evaluator import evaluate
from lm_eval.models.utils import Collator, postprocess_generated_text


def _request(context: str, doc_id: int = 0, gen_kwargs: dict | None = None) -> Instance:
    return Instance(
        request_type="generate_until",
        doc={"context": context},
        arguments=(context, {} if gen_kwargs is None else gen_kwargs),
        idx=0,
        metadata=("generation_test", doc_id, 1),
    )


class _RecordingGenerationLM(LM):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def generate_until(self, requests: list[Instance]) -> list[str]:
        self.calls.append([request.args[0] for request in requests])
        results = []
        for request in requests:
            context = request.args[0]
            result = GenerationResult(f"processed:{context}", f"raw:{context}")
            self.cache_hook.add_partial("generate_until", request.args, result)
            results.append(result)
        return results

    def loglikelihood(self, requests: list[Instance]):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError


class _MultiFilterTask:
    VERSION = 1
    OUTPUT_TYPE = "generate_until"
    UNSAFE_CODE = False

    def __init__(self) -> None:
        self._metric_fn_list = {"score": mean}
        self._docs = [{"context": "prompt", "target": "answer"}]
        self.instances: list[Instance] = []

    @property
    def task_name(self) -> str:
        return "generation_test"

    @property
    def eval_docs(self) -> list[dict[str, str]]:
        return self._docs

    def build_all_requests(self, **kwargs) -> None:
        self.instances = [_request("prompt")]

    def apply_filters(self) -> None:
        for instance in self.instances:
            response = instance.resps[0]
            instance.filtered_resps["first"] = response.removeprefix("processed:")
            instance.filtered_resps["second"] = response.upper()

    def doc_iterator(self, **kwargs):
        yield 0, self._docs[0]

    def process_results(self, doc, results):
        return {"score": float(results[0] in {"prompt", "PROCESSED:PROMPT"})}

    def doc_to_target(self, doc) -> str:
        return doc["target"]

    def aggregation(self):
        return {"score": mean}

    def higher_is_better(self):
        return {"score": True}

    def dump_config(self):
        return {}


def test_instance_keeps_consecutive_generation_pairs_aligned():
    request = _request("prompt")

    request.append_response("processed:first", raw_response="raw:first")
    request.append_response("processed:second", raw_response="raw:second")

    assert request.resps == ["processed:first", "processed:second"]
    assert request.raw_resps == ["raw:first", "raw:second"]
    assert request.resps_for_logging == ["raw:first", "raw:second"]


def test_instance_preserves_existing_positional_dataclass_api():
    request = Instance(
        "generate_until",
        {"context": "prompt"},
        ("prompt", {}),
        0,
        ("generation_test", 0, 1),
        ["processed:prompt"],
        {"none": "prompt"},
        "ignored-task-name",
        99,
        99,
    )

    assert request.resps == ["processed:prompt"]
    assert request.raw_resps == []
    assert request.filtered_resps == {"none": "prompt"}
    assert "raw_resps" not in repr(request)
    assert (request.task_name, request.doc_id, request.repeats) == (
        "generation_test",
        0,
        1,
    )


@pytest.mark.parametrize("missing_raw_field", [False, True])
def test_instance_aligns_raw_responses_with_legacy_state(missing_raw_field):
    request = _request("prompt")
    request.resps.append("processed:legacy")
    if missing_raw_field:
        del request.raw_resps

    assert request.resps_for_logging == ["processed:legacy"]

    request.append_response("processed:new", raw_response="raw:new")

    assert request.raw_resps == [None, "raw:new"]
    assert request.resps_for_logging == ["processed:legacy", "raw:new"]


def test_generation_result_survives_collator_reordering():
    contexts = ["short", "a much longer context", "medium length"]
    collator = Collator(contexts, sort_fn=lambda context: -len(context))
    ordered_contexts = [
        context for batch in collator.get_batched(n=2) for context in batch
    ]
    ordered_results = [
        GenerationResult(f"processed:{context}", f"raw:{context}")
        for context in ordered_contexts
    ]

    restored = collator.get_original(ordered_results)

    assert [result.processed for result in restored] == [
        f"processed:{context}" for context in contexts
    ]
    assert [result.raw for result in restored] == [
        f"raw:{context}" for context in contexts
    ]


@pytest.mark.parametrize("think_end_token", [99, "</think>"])
def test_hflm_preserves_raw_text_before_thinking_and_stop_postprocessing(
    think_end_token,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from lm_eval.models.huggingface import HFLM

    model = HFLM.__new__(HFLM)
    LM.__init__(model)
    model.backend = "causal"
    model.batch_size_per_gpu = 1
    model.think_end_token = think_end_token
    model.truncation = False
    model._device = torch.device("cpu")
    model._max_length = 32
    model.tokenizer = SimpleNamespace(eos_token_id=0)

    pieces = {
        0: "<eos>",
        10: "reason",
        99: "</think>",
        32: " ",
        42: "answer",
        88: "STOP",
    }

    def decode(tokens, **kwargs):
        if isinstance(tokens, int):
            return pieces[tokens]
        return "".join(pieces[token] for token in tokens)

    model.tok_encode = lambda text: [1] * len(text)
    model.tok_decode = decode
    model.tok_batch_encode = lambda contexts, **kwargs: (
        torch.tensor([[1, 2] for _ in contexts]),
        torch.tensor([[1, 1] for _ in contexts]),
    )
    generation_stops = []

    def generate(**kwargs):
        generation_stops.append(kwargs["stop"])
        return torch.tensor([[1, 2, 10, 99, 32, 42, 88]])

    model._model_generate = generate
    result = model.generate_until(
        [_request("ctx", gen_kwargs={"until": ["STOP"], "max_gen_toks": 7})],
        disable_tqdm=True,
    )[0]

    assert isinstance(result, GenerationResult)
    assert result.processed == "answer"
    assert result.raw == "reason</think> answerSTOP"
    assert generation_stops == [["STOP", "<eos>"]]


def test_vllm_postprocess_contract_keeps_preprocessed_generation_available():
    raw = "reasoning STOP inside trace</think> answerSTOPignored"

    processed = postprocess_generated_text(
        generation=raw,
        stop=["STOP"],
        think_end_token="</think>",  # noqa: S106
    )
    result = GenerationResult(processed, raw)

    assert result.processed == "answer"
    assert result.raw == raw


def test_generation_result_stays_aligned_across_mixed_and_full_cache_hits(tmp_path):
    model = _RecordingGenerationLM()
    cached = CachingLM(model, str(tmp_path / "responses.sqlite"))

    try:
        first = cached.generate_until([_request("a")])
        mixed = cached.generate_until([_request("a"), _request("b", doc_id=1)])
        full = cached.generate_until([_request("a"), _request("b", doc_id=1)])
    finally:
        cached.dbdict.close()

    assert model.calls == [["a"], ["b"]]
    assert [
        [result.processed for result in batch] for batch in (first, mixed, full)
    ] == [
        ["processed:a"],
        ["processed:a", "processed:b"],
        ["processed:a", "processed:b"],
    ]
    assert [[result.raw for result in batch] for batch in (first, mixed, full)] == [
        ["raw:a"],
        ["raw:a", "raw:b"],
        ["raw:a", "raw:b"],
    ]


def test_legacy_string_cache_hit_falls_back_without_misaligning_new_raw_result(
    tmp_path,
):
    model = _RecordingGenerationLM()
    cached = CachingLM(model, str(tmp_path / "responses.sqlite"))
    request_a = _request("a")
    request_b = _request("b", doc_id=1)
    cached.dbdict[hash_args("generate_until", request_a.args)] = "processed:a"

    try:
        results = cached.generate_until([request_a, request_b])
    finally:
        cached.dbdict.close()

    for response, request in zip(results, (request_a, request_b), strict=True):
        raw_response = response.raw if isinstance(response, GenerationResult) else None
        request.append_response(str(response), raw_response=raw_response)

    assert model.calls == [["b"]]
    assert request_a.resps_for_logging == ["processed:a"]
    assert request_b.resps_for_logging == ["raw:b"]


def test_evaluator_keeps_multi_filter_metrics_and_raw_logs_independent():
    task = _MultiFilterTask()
    results = evaluate(
        lm=_RecordingGenerationLM(),
        task_dict={"tasks": {task.task_name: task}, "groups": {}},
        bootstrap_iters=0,
        log_samples=True,
    )

    assert results is not None
    assert results["results"][task.task_name]["score,first"] == 1.0
    assert results["results"][task.task_name]["score,second"] == 1.0

    samples = results["samples"][task.task_name]
    assert [sample["filter"] for sample in samples] == ["first", "second"]
    assert [sample["resps"] for sample in samples] == [[["raw:prompt"]]] * 2
    assert [sample["filtered_resps"] for sample in samples] == [
        ["prompt"],
        ["PROCESSED:PROMPT"],
    ]
