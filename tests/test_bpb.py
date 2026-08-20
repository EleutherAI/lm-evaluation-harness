import math

import pytest

import lm_eval.api.task as task_module
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import bpb_corpus, bpb_macro
from lm_eval.api.model import TemplateLM
from lm_eval.api.task import ConfigurableTask, MultipleChoiceTask
from lm_eval.config.task import TaskConfig
from lm_eval.evaluator import (
    _request_type_padding,
    _request_types_to_execute,
    _synthetic_padding_request,
)
from lm_eval.tasks.bbh.cot_fewshot import utils as bbh_utils
from lm_eval.tasks.gsm8k import utils as gsm8k_utils


class CharacterTemplateLM(TemplateLM):
    backend = "causal"
    prefix_token_id = 0

    @property
    def eot_token_id(self):
        return 0

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return [ord(character) for character in string]

    def _loglikelihood_tokens(self, requests, **kwargs):
        return [(-float(len(continuation)), False) for _, _, continuation in requests]

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError


class EmptyLegacyTask(MultipleChoiceTask):
    DATASET_PATH = None

    def download(self, *args, **kwargs):
        self.dataset = {}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return []

    def doc_to_text(self, doc):
        return doc["question"]


def make_task(**config_overrides):
    config = {
        "task": "test_bpb",
        "output_type": "multiple_choice",
        "doc_to_choice": ["A", "é"],
        "doc_to_target": 1,
        "target_delimiter": " ",
        "metric_list": [{"metric": "acc"}],
    }
    config.update(config_overrides)
    task = object.__new__(ConfigurableTask)
    task._config = TaskConfig(**config)
    task.OUTPUT_TYPE = task.config.output_type
    task.prompt = None
    task.features = {}
    task.multiple_input = 0
    task.multiple_target = 0
    task._metric_fn_list = {
        metric["metric"]: None for metric in task.config.metric_list
    }
    task._metric_fn_kwargs = {
        metric["metric"]: {} for metric in task.config.metric_list
    }
    task._aggregation_list = {}
    task._higher_is_better = {}
    task._compute_bpb = True
    return task


def test_bpb_aggregations_distinguish_macro_and_corpus_weighting():
    # One short, easy sample and one long, hard sample. Macro gives the two
    # examples equal weight; corpus BPB pools their likelihood and byte counts.
    samples = [(-math.log(2), 1), (-12 * math.log(2), 4)]
    per_example = [-ll / (num_bytes * math.log(2)) for ll, num_bytes in samples]

    assert bpb_macro(per_example) == pytest.approx(2.0)
    assert bpb_corpus(samples) == pytest.approx(13 / 5)


def test_template_backend_records_exact_scored_continuation_tokens():
    model = CharacterTemplateLM()
    request = Instance("loglikelihood", {}, ("context ", "é"), 0)

    result = model.loglikelihood([request], disable_tqdm=True)

    # TemplateLM moves the context's trailing space into the continuation;
    # request metadata must describe the exact two tokens actually scored.
    assert result == [(-2.0, False)]
    assert request.continuation_token_count == 2


def test_template_backend_handles_an_empty_continuation_without_indexing_it():
    model = CharacterTemplateLM()
    request = Instance("loglikelihood", {}, ("", ""), 0)

    assert model.loglikelihood([request], disable_tqdm=True) == [(-0.0, False)]
    assert request.continuation_token_count == 0


def test_multiple_choice_bpb_scores_gold_continuation_in_utf8_bytes():
    task = make_task()
    instances = [
        Instance("loglikelihood", {}, ("prompt", " A"), 0),
        Instance("loglikelihood", {}, ("prompt", " é"), 1),
    ]
    results = [(-0.2, False), (-3 * math.log(2), False)]

    metrics = task._compute_conditional_bpb({}, results, instances)

    # The scored continuation is one ASCII space plus a two-byte UTF-8 letter.
    assert metrics["bpb_macro"] == pytest.approx(1.0)
    assert metrics["bits_per_byte_corr"] == metrics["bpb_macro"]
    assert metrics["bpb_corpus"] == pytest.approx((-3 * math.log(2), 3))
    assert metrics["bpb_total_loglikelihood"] == pytest.approx(-3 * math.log(2))
    assert metrics["bpb_total_bytes"] == 3


def test_acc_per_token_uses_model_token_counts_not_character_lengths():
    task = make_task(
        doc_to_choice=["long", "x"],
        metric_list=[{"metric": "acc"}, {"metric": "acc_per_token"}],
    )
    instances = [
        Instance("loglikelihood", {}, ("prompt", " long"), 0),
        Instance("loglikelihood", {}, ("prompt", " x"), 1),
    ]
    instances[0].continuation_token_count = 4
    instances[1].continuation_token_count = 1
    results = [(-2.0, False), (-1.0, False)]

    metrics = task.process_results_with_instances({}, results, instances)

    assert metrics["acc"] == 1.0
    assert metrics["acc_per_token"] == 0.0


def test_generate_until_adds_one_teacher_forced_gold_request():
    task = make_task(
        output_type="generate_until",
        doc_to_choice=None,
        doc_to_target="answer",
        doc_to_bpb_target="gold",
        bpb_target_delimiter="",
        generation_kwargs={"until": ["stop"]},
        metric_list=[{"metric": "exact_match"}],
    )
    task.features = {"answer": object(), "gold": object()}

    requests = task.construct_requests(
        {"answer": "generated", "gold": "canonical"},
        "prompt",
        metadata=("test_bpb", 0, 3),
    )

    assert len(requests) == 2
    primary, auxiliary = requests
    assert primary.request_type == "generate_until"
    assert primary.repeats == 3
    assert auxiliary.request_type == "loglikelihood"
    assert auxiliary.args == ("prompt", "canonical")
    assert auxiliary.repeats == 1
    assert auxiliary.is_bpb_auxiliary is True


def test_generate_until_defaults_to_olmes_leading_space():
    task = make_task(
        output_type="generate_until",
        doc_to_choice=None,
        doc_to_target="answer",
        doc_to_bpb_target="gold",
        generation_kwargs={"until": ["stop"]},
        metric_list=[{"metric": "exact_match"}],
    )
    task.features = {"answer": object(), "gold": object()}

    requests = task.construct_requests(
        {"answer": "generated", "gold": "canonical"},
        "prompt",
        metadata=("test_bpb", 0, 1),
    )

    assert requests[1].args == ("prompt", " canonical")


def test_request_cache_key_isolated_by_bpb_schema(monkeypatch):
    task = make_task()
    captured = []
    cached = Instance("loglikelihood", {}, ("prompt", " answer"), 0)

    def fake_load(*, file_name, cache):
        captured.append(file_name)
        return [[cached]]

    monkeypatch.setattr(task_module, "load_from_cache", fake_load)
    task.build_all_requests(cache_requests=True, tokenizer_name="tokenizer")

    assert "request_schema2-bpb1" in captured[0]


def test_direct_task_subclasses_default_to_bpb_disabled(monkeypatch):
    task = EmptyLegacyTask(config={"task": "empty_legacy", "num_fewshot": 0})
    cached = Instance("loglikelihood", {}, ("prompt", " answer"), 0)
    monkeypatch.setattr(task_module, "load_from_cache", lambda **kwargs: [[cached]])

    task.build_all_requests(cache_requests=True)

    assert task._compute_bpb is False
    assert task.instances == [cached]


def test_distributed_padding_is_computed_per_request_type_after_repeats():
    import torch

    instances = [
        Instance("generate_until", {}, ("prompt", {}), 0, metadata=("task", 0, 2)),
        Instance(
            "loglikelihood",
            {},
            ("prompt", " answer"),
            0,
            metadata=("task", 0, 1),
        ),
    ]

    class FakeDistributedLM:
        world_size = 2
        rank = 0
        device = "cpu"

        def all_gather(self, counts):
            # Both ranks have three total requests, but with opposing method
            # counts. Aggregate-only padding would incorrectly return zero.
            assert counts.tolist() == [1, 0, 2]
            return torch.tensor([[1, 0, 2], [2, 0, 1]])

    padding = _request_type_padding(instances, FakeDistributedLM())

    assert padding == {
        "loglikelihood": 1,
        "loglikelihood_rolling": 0,
        "generate_until": 0,
    }


def test_distributed_rank_executes_and_can_pad_a_request_type_absent_locally():
    requests = {"generate_until": [object()]}
    padding = {
        "loglikelihood": 2,
        "loglikelihood_rolling": 0,
        "generate_until": 0,
    }

    assert _request_types_to_execute(requests, padding) == (
        "loglikelihood",
        "generate_until",
    )
    dummy = _synthetic_padding_request("loglikelihood")
    assert dummy.request_type == "loglikelihood"
    assert dummy.repeats == 1
    assert dummy.args == (" ", " x")


def test_generate_until_can_override_the_bpb_context():
    task = make_task(
        output_type="generate_until",
        doc_to_choice=None,
        doc_to_text="bpb_prompt",
        doc_to_target="answer",
        doc_to_bpb_target="gold",
        doc_to_bpb_text="bpb_prompt",
        generation_kwargs={"until": ["stop"]},
        metric_list=[{"metric": "exact_match"}],
    )
    task.features = {
        "answer": object(),
        "gold": object(),
        "bpb_prompt": object(),
    }

    requests = task.construct_requests(
        {"answer": "generated", "gold": "canonical", "bpb_prompt": "short"},
        "full few-shot prompt",
        metadata=("test_bpb", 0, 1),
    )

    assert requests[1].args == ("short", " canonical")


def test_enable_bpb_rejects_tasks_without_a_gold_continuation():
    task = make_task(bpb_unsupported_reason="the source has no reference completion")

    with pytest.raises(ValueError, match="source has no reference completion"):
        task.enable_bpb()


def test_gsm8k_bpb_target_matches_olmes_naturalization():
    target = gsm8k_utils.bpb_target(
        {"answer": ("She has 3+4 = <<3+4=7>>7 apples. #### 7")}
    )

    assert target == "She has 3 + 4 = 7 apples. So the answer is 7."


def test_bbh_bpb_uses_the_short_non_cot_query_and_answer():
    doc = {
        "input": "True and not False is",
        "target": "Reasoning here. So the answer is True.",
    }

    assert bbh_utils.bpb_text(doc) == "Q: True and not False is\nA:"
    assert bbh_utils.bpb_target(doc) == "True"


def test_acc_per_token_requires_runtime_token_counts():
    task = make_task(metric_list=[{"metric": "acc_per_token"}])
    instances = [
        Instance("loglikelihood", {}, ("prompt", " A"), 0),
        Instance("loglikelihood", {}, ("prompt", " é"), 1),
    ]

    with pytest.raises(ValueError, match="continuation token counts"):
        task.process_results_with_instances(
            {}, [(-2.0, False), (-1.0, False)], instances
        )
