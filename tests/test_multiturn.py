"""Offline regressions for sequential generation through the normal evaluator."""

from copy import deepcopy

import datasets
import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.multiturn import run_multi_turn_rollout
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig
from lm_eval.evaluator import evaluate


class RecordingLM(LM):
    def __init__(self):
        super().__init__()
        self.waves = []

    def loglikelihood(self, requests):
        raise AssertionError("Unexpected method")

    def loglikelihood_rolling(self, requests):
        raise AssertionError("Unexpected method")

    def generate_until(self, requests):
        self.waves.append([deepcopy(req.args) for req in requests])
        return ["reply:" + req.args[0] for req in requests]


class EpisodeTask(ConfigurableTask):
    def __init__(self, docs, name="episodes", repeats=1, **kwargs):
        self.docs = docs
        super().__init__(
            config={
                "task": name,
                "output_type": "multi_turn_generate",
                "test_split": "test",
                "num_fewshot": 0,
                "doc_to_target": "",
                "repeats": repeats,
                "generation_kwargs": {"until": [], "max_gen_toks": 16},
                "metric_list": [
                    {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
                ],
            },
            **kwargs,
        )

    def download(self, dataset_kwargs=None, **kwargs):
        self.dataset = {"test": datasets.Dataset.from_list(self.docs)}

    def doc_to_text(self, doc):
        return doc["name"]

    def init_multiturn_state(self, doc, ctx, gen_kwargs, **kwargs):
        return {"turn": 0, "answers": [], "doc": doc, "kwargs": gen_kwargs}

    def multiturn_is_done(self, state):
        return state["turn"] >= state["doc"]["turns"]

    def multiturn_next_request(self, state):
        # Each subsequent prompt must contain the model's actual previous answer.
        prompt = state["answers"][-1] if state["answers"] else state["doc"]["name"]
        return prompt, deepcopy(state["kwargs"])

    def multiturn_consume_response(self, state, response):
        state["answers"].append(response)
        state["turn"] += 1

    def multiturn_result(self, state):
        assert self.multiturn_is_done(state)
        return state["answers"]

    def process_results(self, doc, results):
        expected = []
        answer = doc["name"]
        for _ in range(doc["turns"]):
            answer = "reply:" + answer
            expected.append(answer)
        return {"acc": float(all(result == expected for result in results))}


def evaluate_tasks(lm, *tasks, **kwargs):
    return evaluate(
        lm=lm,
        task_dict={t.config.task: t for t in tasks},
        bootstrap_iters=0,
        log_samples=True,
        **kwargs,
    )


def test_wave_batching_history_early_completion_and_independent_repeats():
    task = EpisodeTask(
        [
            {"name": "zero", "turns": 0},
            {"name": "short", "turns": 1},
            {"name": "long", "turns": 3},
        ],
        repeats=2,
    )
    lm = RecordingLM()
    result = evaluate_tasks(lm, task)
    assert [len(wave) for wave in lm.waves] == [4, 2, 2]
    assert result["results"]["episodes"]["acc,none"] == 1
    samples = result["samples"]["episodes"]
    assert len(samples) == 3
    assert all(len(row["resps"][0]) == 2 for row in samples)
    assert all(kwargs["max_gen_toks"] == 16 for wave in lm.waves for _, kwargs in wave)


def test_mixed_single_turn_and_multiturn_evaluation():
    multi = EpisodeTask([{"name": "multi", "turns": 2}])
    single = EpisodeTask([{"name": "single", "turns": 1}], name="single")
    single.OUTPUT_TYPE = "generate_until"
    single.set_config("output_type", "generate_until")
    single.process_results = lambda doc, results: {
        "acc": float(results == ["reply:single"])
    }
    lm = RecordingLM()
    result = evaluate_tasks(lm, multi, single)
    assert result["results"]["episodes"]["acc,none"] == 1
    assert result["results"]["single"]["acc,none"] == 1
    assert len(lm.waves) == 3


def placeholder(task):
    return Instance(
        request_type="multi_turn_generate",
        doc=task.docs[0],
        arguments=("", {}),
        idx=0,
        metadata=(task.config.task, 0, 1),
    )


def test_step_caps_are_per_task_and_never_score_partial_results():
    short = EpisodeTask([{"name": "short", "turns": 1}], name="short")
    short.MAX_MULTITURN_STEPS = 1
    long = EpisodeTask([{"name": "long", "turns": 3}], name="long")
    long.MAX_MULTITURN_STEPS = 3
    assert (
        evaluate_tasks(RecordingLM(), short, long)["results"]["long"]["acc,none"] == 1
    )
    short.docs[0]["turns"] = 2
    req = placeholder(short)
    with pytest.raises(RuntimeError, match="step cap"):
        run_multi_turn_rollout(RecordingLM(), {"short": short}, [req])
    assert not req.resps


def test_none_requires_explicit_completion():
    task = EpisodeTask([{"name": "stalled", "turns": 1}])
    task.multiturn_next_request = lambda state: None
    with pytest.raises(RuntimeError, match="not done"):
        evaluate_tasks(RecordingLM(), task)


@pytest.mark.parametrize("delta", [-1, 1])
def test_wrong_response_count_fails(delta):
    task = EpisodeTask([{"name": "count", "turns": 1}])
    lm = RecordingLM()
    lm.generate_until = lambda requests: ["x"] * (len(requests) + delta)
    with pytest.raises(ValueError, match="zip"):
        evaluate_tasks(lm, task)


def test_distributed_execution_fails_before_collectives_or_requests():
    task = EpisodeTask([{"name": "distributed", "turns": 1}])
    lm = RecordingLM()
    lm._world_size = 2
    with pytest.raises(NotImplementedError, match="world_size=1"):
        evaluate_tasks(lm, task)
    assert not lm.waves
    assert getattr(task, "_instances", None) is None


def test_multiturn_generation_defaults():
    cfg = TaskConfig(output_type="multi_turn_generate")
    assert cfg.generation_kwargs["max_gen_toks"] > 0


def test_simple_evaluate_overrides_multiturn_generation_kwargs():
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager

    task = EpisodeTask([{"name": "override", "turns": 2}])
    lm = RecordingLM()
    result = simple_evaluate(
        model=lm,
        tasks=[task],
        task_manager=TaskManager(),
        gen_kwargs={"max_gen_toks": 7},
        bootstrap_iters=0,
    )
    assert result["results"]["episodes"]["acc,none"] == 1
    assert all(kwargs["max_gen_toks"] == 7 for wave in lm.waves for _, kwargs in wave)


def test_backend_errors_propagate_without_episode_results():
    task = EpisodeTask([{"name": "failure", "turns": 2}])
    req = placeholder(task)
    lm = RecordingLM()

    def fail(requests):
        raise RuntimeError("Backend unavailable")

    lm.generate_until = fail
    with pytest.raises(RuntimeError, match="Backend unavailable"):
        run_multi_turn_rollout(lm, {task.config.task: task}, [req])
    assert not req.resps
