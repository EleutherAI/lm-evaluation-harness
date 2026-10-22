"""Contract tests for frozen rollouts and offline judging (no GPU/model downloads)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lm_eval.judging.core import (RUBRIC, atomic_json, collect_verdicts, digest,
    freeze_rollouts, indexed, judge_inputs, native_verdict, parse_verdict, prepare,
    read_jsonl, summarize, write_jsonl)
from lm_eval.judging.runner import balanced_smoke, inference_command, read_partial
from lm_eval.models.cached_rollouts import CachedRollouts


@pytest.fixture
def run(tmp_path):
    doc = {"id": "a", "prompt": "Question?", "reference": "Answer.",
           "language": "French", "category": "Wordplay"}
    doc["input_hash"] = digest(doc)
    write_jsonl(tmp_path / "dataset.jsonl", [doc])
    atomic_json(tmp_path / "manifest.json", {"generation_hash": "generation", "count": 1})
    freeze_rollouts(tmp_path, [{"id": "a", "input_hash": doc["input_hash"],
        "output": "Answer.", "reasoning_content": "private reasoning",
        "inference_status": "ok", "finish_reason": "stop"}])
    return tmp_path


def test_prepare_offline_and_no_reference_leakage(tmp_path):
    from datasets import Dataset, DatasetDict

    source = tmp_path / "source"
    DatasetDict(test=Dataset.from_list([{"task_id": "a", "i18n_prompt": "question",
        "i18n_gtfa": "SECRET_REFERENCE", "language": "Chinese", "category": "Math"}])).save_to_disk(source)
    out = tmp_path / "run"
    manifest = prepare(source, out, {"temperature": 0, "seed": 1234})
    assert manifest["count"] == 1
    assert "SECRET_REFERENCE" not in (out / "generation-input.jsonl").read_text()
    assert prepare(source, out, {"temperature": 0, "seed": 1234}) == manifest
    with pytest.raises(ValueError):
        prepare(source, out, {"temperature": 1, "seed": 1234})


def test_frozen_answer_and_judge_trace_isolation(run):
    requests = judge_inputs(run, {"model": "judge"})
    assert "private reasoning" not in json.dumps(requests)
    assert requests[0]["state"]["candidate_answer"] == "Answer."
    assert requests[0]["rubric_hash"] == digest(RUBRIC)
    with pytest.raises(ValueError):
        write_jsonl(run / "rollouts.jsonl", [], immutable=True)


@pytest.mark.parametrize("text", ["yes", "{\"correct\":1}", "1 because", "<think>yes</think>1", "", None])
def test_strict_verdicts(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_verdict_hashes_and_failures(run):
    spec = {"model": "judge"}
    request = judge_inputs(run, spec)[0]
    output = {**request, "output": "1", "inference_status": "ok"}
    rows = collect_verdicts(run, spec, [output])
    assert summarize(read_jsonl(run / "rollouts.jsonl"), rows)["accuracy"] == 1
    broken = {**output, "output": "invalid"}
    result = summarize(read_jsonl(run / "rollouts.jsonl"), collect_verdicts(run, spec, [broken]))
    assert result["failures"] == 1 and result["accuracy"] is None
    for field in ("answer_hash", "judge_hash", "rubric_hash"):
        with pytest.raises(ValueError, match="mismatch"):
            collect_verdicts(run, spec, [{**output, field: "changed"}])
    with pytest.raises(ValueError, match="coverage"):
        collect_verdicts(run, spec, [])


def test_native_choice_probability_contract():
    answer = {"type": "choice", "choice": "correct", "probabilities": {"correct": .8, "incorrect": .2}}
    assert native_verdict(answer) == 1
    with pytest.raises(ValueError):
        native_verdict({**answer, "choice": "incorrect"})
    with pytest.raises(ValueError):
        native_verdict({**answer, "probabilities": {"correct": float("nan"), "incorrect": .2}})


def test_replay_requires_matching_prompt_and_identity(run):
    model = CachedRollouts(run / "rollouts.jsonl")
    doc = read_jsonl(run / "rollouts.jsonl")[0]
    request = SimpleNamespace(doc=doc, args=(doc["prompt"], {}))
    assert model.generate_until([request]) == ["Answer."]
    request.args = ("Changed prompt", {})
    with pytest.raises(ValueError):
        model.generate_until([request])


def test_duplicate_and_partial_output(tmp_path):
    with pytest.raises(ValueError):
        indexed([{"id": "a"}, {"id": "a"}])
    file = tmp_path / "out"
    file.write_text('{"id":"a"}\n{"id":')
    assert read_partial(file) == [{"id": "a"}]
    file.write_text('invalid\n{"id":"a"}\n')
    with pytest.raises(ValueError):
        read_partial(file)


def test_transport_commands_and_balanced_smoke(tmp_path):
    config = {"inference_repo": "/engine", "profiles_dir": "/profiles"}
    spec = {"model": "/weights", "profile": "gpt-oss", "level": "high",
            "max_new_tokens": 8192, "transport": "batch"}
    cmd = inference_command(config, spec, "in", "out", judge=True)
    assert cmd[cmd.index("--replicas-per-node") + 1] == "1"
    assert "--output-validator" in cmd and "--chat-completions" not in cmd
    cmd = inference_command(config, {**spec, "transport": "http"}, "in", "out", judge=False)
    assert "--chat-completions" in cmd and "--output-validator" not in cmd
    docs = [{"id": str(i), "language": lang} for i, lang in enumerate(["French"] * 3 + ["Chinese"] * 3)]
    assert len(balanced_smoke(docs, docs)) == 4


def test_unsupported_reasoning_levels():
    from inference.common.thinking import native_thinking_kwargs

    with pytest.raises(ValueError):
        native_thinking_kwargs("gpt_oss", "off")
    assert native_thinking_kwargs("on_off", "off") == {"enable_thinking": False}


def test_harness_task_matches_artifact(run):
    from lm_eval.tasks.multinrc.utils import load_dataset, process_results

    spec = {"id": "judge", "model": "test"}
    atomic_json(run / "config.json", {"stages": [{"id": "rollout"}, spec]})
    rows = collect_verdicts(run, spec, [{**judge_inputs(run, spec)[0], "output": "1", "inference_status": "ok"}])
    write_jsonl(run / "judge/verdicts.jsonl", rows)
    doc = load_dataset(run=str(run), judge_id="judge")["test"][0]
    assert process_results(doc, ["Answer."]) == {"acc": 1.0}
    with pytest.raises(ValueError):
        process_results(doc, ["different answer"])


def test_harness_evaluator_replays_and_aggregates(run, monkeypatch):
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    spec = {"id": "judge", "model": "test"}
    atomic_json(run / "config.json", {"stages": [{"id": "rollout"}, spec]})
    rows = collect_verdicts(run, spec, [{**judge_inputs(run, spec)[0], "output": "1", "inference_status": "ok"}])
    write_jsonl(run / "judge/verdicts.jsonl", rows)
    monkeypatch.setenv("MULTINRC_RUN", str(run))
    monkeypatch.setenv("MULTINRC_JUDGE", "judge")
    tasks = Path(__file__).parents[1] / "lm_eval/tasks/multinrc"
    manager = TaskManager(include_defaults=False, include_path=str(tasks))
    result = evaluator.simple_evaluate(model=CachedRollouts(run / "rollouts.jsonl"),
        tasks=["multinrc_judged"], task_manager=manager, num_fewshot=0,
        log_samples=True, bootstrap_iters=0)
    assert result["results"]["multinrc_judged"]["acc,none"] == 1.0
    assert len(result["samples"]["multinrc_judged"]) == 1


def test_stage_smoke_resume_and_frozen_outputs(tmp_path, monkeypatch):
    from lm_eval.judging.runner import execute

    doc = {"id": "a", "prompt": "Q", "reference": "A", "language": "French", "category": "Math"}
    doc["input_hash"] = digest(doc)
    write_jsonl(tmp_path / "dataset.jsonl", [doc])
    write_jsonl(tmp_path / "generation-input.jsonl", [{"id": "a", "input_hash": doc["input_hash"],
        "messages": [{"role": "user", "content": "Q"}]}])
    atomic_json(tmp_path / "manifest.json", {"count": 1, "generation_hash": "g"})
    config = {"output": str(tmp_path), "inference_repo": "/engine", "harness": "/harness",
              "profiles_dir": "/profiles", "metadata": "/metadata", "stages": [
                  {"id": "rollout", "model": "generator", "profile": "qwen", "level": "high",
                   "max_new_tokens": 16, "transport": "http"},
                  {"id": "judge", "model": "judge", "profile": "qwen", "level": "off",
                   "max_new_tokens": 16, "transport": "http"}]}
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    calls = []
    def fake_process(command, log, timeout, env):
        source = command[command.index("--input_file") + 1]
        target = command[command.index("--output_file") + 1]
        inputs = read_jsonl(source)
        calls.extend(row["id"] for row in inputs)
        judged = "--output-validator" in command
        write_jsonl(target, [{**row, "output": "1" if judged else "A",
                             "inference_status": "ok"} for row in inputs])
        return {"returncode": 0, "timed_out": False, "elapsed_seconds": .01}
    monkeypatch.setattr("lm_eval.judging.runner.run_process", fake_process)
    assert execute(config, 0)["complete"]
    assert execute(config, 0)["new_rows"] == 0
    assert calls == ["a"]
    assert execute(config, 1)["complete"]
    assert json.loads((tmp_path / "judge/summary.json").read_text())["accuracy"] == 1


def test_empty_truncated_answer_is_preserved_and_never_replaced_with_trace(run):
    doc = read_jsonl(run / "dataset.jsonl")[0]
    (run / "rollouts.jsonl").unlink()
    rows = freeze_rollouts(run, [{"id": "a", "input_hash": doc["input_hash"], "output": "",
        "reasoning_content": "unfinished private reasoning", "inference_status": "discarded_max_length"}])
    assert rows[0]["empty_answer"]
    assert judge_inputs(run, {"model": "judge"})[0]["state"]["candidate_answer"] == ""
