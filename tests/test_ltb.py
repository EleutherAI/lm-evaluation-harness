"""Offline regression tests for the Last Translation Benchmark integration."""

import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from lm_eval.tasks.ltb import utils


@pytest.fixture(autouse=True)
def isolated_judge_output(monkeypatch, tmp_path):
    monkeypatch.setenv("LTB_OUTPUT_DIR", str(tmp_path / "audit"))


@pytest.fixture
def doc():
    return {
        "id": 337,
        "source_text": "Our new nurse used to play in the men's professional football team.",
        "source_lang": "English",
        "target_lang": "German",
        "source_media": None,
        "source_instructions": None,
        "tags": ["LTBv1", "LTBv1-eval"],
        "verification_rules": [
            "Nurse must be masculine.",
            "Preserve the football reference.",
        ],
        "translations": [{"model": "human", "translation": "SECRET REFERENCE"}],
    }


def write_data(tmp_path, records):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "<|channel|>analysis<|message|>Reasoning<|end|><|start|>assistant<|channel|>final<|message|>Übersetzung.<|return|>",
            "Übersetzung.",
        ),
        (
            "<|channel|>final<|message|>Bonjour.\nDeuxième ligne.",
            "Bonjour.\nDeuxième ligne.",
        ),
        (
            "<|meta_sep|>analysis<|im_sep|>Reasoning<|im_end|><|im_start|>assistant<|meta_sep|>final<|im_sep|>Hello.<|fim_suffix|>",
            "Hello.",
        ),
        ("<|channel|>analysis<|message|>Unfinished reasoning", None),
        ("", None),
    ],
)
def test_harmony_judges_only_final_channel(monkeypatch, doc, raw, expected):
    monkeypatch.setenv("LTB_RESPONSE_FORMAT", "harmony")
    results = [raw]
    payload = utils.process_results(doc, results)["ltb_pass_rate"]
    assert payload["translation"] == expected
    assert results == [raw]  # Preserve the original response for sample logging.


def test_plain_ltb_preserves_translation(monkeypatch, doc):
    monkeypatch.delenv("LTB_RESPONSE_FORMAT", raising=False)
    raw = "  A translation.\n"
    assert utils.process_results(doc, [raw])["ltb_pass_rate"]["translation"] == raw


@pytest.mark.parametrize("response_format", ["text", "harmony"])
@pytest.mark.parametrize(
    "raw", [None, 42, {"error": "failed"}, ["translation"], b"text"]
)
def test_non_string_generations_fail_and_export_null(
    monkeypatch, doc, response_format, raw
):
    monkeypatch.setenv("LTB_RESPONSE_FORMAT", response_format)
    results = [raw]
    item = utils.process_results(doc, results)["ltb_pass_rate"]
    assert item["translation"] is None
    assert results[0] is raw
    client = MagicMock()
    with utils.JudgeLog([item]) as audit:
        assert utils._score_example(client, "judge", item, audit) == 0
        assert json.loads((audit.directory / "ltb_submission.json").read_text()) == [
            {"id": doc["id"], "translation": None}
        ]
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("raw", [None, "translation"])
def test_invalid_response_format_rejected(monkeypatch, doc, raw):
    monkeypatch.setenv("LTB_RESPONSE_FORMAT", "invalid")
    with pytest.raises(ValueError, match="Unsupported LTB_RESPONSE_FORMAT"):
        utils.process_results(doc, [raw])


def test_official_subset_only(doc):
    data = utils.process_docs(
        Dataset.from_list(
            [doc, {**doc, "id": 338, "tags": ["LTBv1"], "source_media": "image"}]
        )
    )
    assert len(data) == 1
    assert data[0]["id"] == 337
    assert "translations" not in data.column_names
    assert "Nurse must" not in utils.doc_to_text(data[0])


@pytest.mark.parametrize(
    "change",
    [
        {"verification_rules": []},
        {"verification_rules": [""]},
        {"source_media": "image"},
        {"source_instructions": "extra instructions"},
    ],
)
def test_reject_malformed_eval_record(doc, change):
    with pytest.raises(ValueError):
        utils.process_docs(Dataset.from_list([{**doc, **change}]))


def test_reject_empty_subset_and_duplicate_ids(doc):
    for records in [[], [doc, doc]]:
        with pytest.raises(ValueError):
            utils.process_docs(Dataset.from_list(records))


def test_upstream_prompts(doc):
    assert utils.doc_to_text(doc) == (
        "Translate the following text from English to German. "
        "Output only the translation and nothing else:\n" + doc["source_text"]
    )
    assert utils.verification_prompt("source", "translation", "rule") == (
        "Your goal is to verify whether a translation fulfills a criterion.\n\n"
        "Criterion: rule\n\nInput: source\n\nTranslation to verify: translation\n\n"
        "Output only pass or fail and nothing else."
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("pass", True),
        ("**PASS**.", True),
        ("It could pass, but fail", False),
        ("It could fail, but pass", True),
        ("PASS because it is correct", True),
        ("fail", False),
        ("unknown", False),
        ("", False),
        (None, False),
    ],
)
def test_official_verdict_parsing(response, expected):
    assert utils.parse_verdict(response) is expected


def response(content, finish_reason="stop"):
    return SimpleNamespace(
        model_dump=lambda **kwargs: {
            "choices": [
                {
                    "message": {"content": content, "reasoning": "judge reasoning"},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"completion_tokens": 3},
        },
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
    )


def test_all_rules_must_pass_and_empty_output_fails(doc):
    item = utils.process_results(doc, ["translation"])["ltb_pass_rate"]
    client = MagicMock()
    client.chat.completions.create.side_effect = [response("pass"), response("fail")]
    assert utils._score_example(client, "judge", item) == 0
    assert client.chat.completions.create.call_count == 2
    client.chat.completions.create.side_effect = [response("pass"), response("pass")]
    assert utils._score_example(client, "judge", item) == 1
    client.reset_mock()
    assert utils._score_example(client, "judge", {**item, "translation": ""}) == 0
    client.chat.completions.create.assert_not_called()


def test_judge_errors_propagate(doc):
    item = utils.process_results(doc, ["translation"])["ltb_pass_rate"]
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        utils._score_example(client, "judge", item)
    client.chat.completions.create.side_effect = None
    client.chat.completions.create.return_value = response("pass", "length")
    with pytest.raises(RuntimeError, match="truncated"):
        utils._score_example(client, "judge", item)


def test_aggregation_uses_example_denominator(monkeypatch, doc):
    monkeypatch.setenv("LTB_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LTB_JUDGE_WORKERS", "1")
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    client = constructor.return_value.__enter__.return_value
    client.chat.completions.create.side_effect = [
        response("pass"),
        response("pass"),
        response("fail"),
    ]
    items = [
        utils.process_results(doc, ["translation"])["ltb_pass_rate"],
        utils.process_results({**doc, "verification_rules": ["one rule"]}, ["bad"])[
            "ltb_pass_rate"
        ],
        utils.process_results(doc, [""])["ltb_pass_rate"],
    ]
    assert utils.aggregate_pass_rate(items) == pytest.approx(1 / 3)


def test_missing_credentials_fail_explicitly(monkeypatch):
    monkeypatch.delenv("LTB_JUDGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LTB_JUDGE_API_KEY"):
        utils.aggregate_pass_rate([{}])


def test_missing_judge_dependency_explained(monkeypatch):
    monkeypatch.setenv("LTB_JUDGE_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match="Install the LTB judge dependency"):
        utils.aggregate_pass_rate([{}])


def test_aggregation_completes_multiple_batches(monkeypatch, doc):
    monkeypatch.setenv("LTB_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LTB_JUDGE_WORKERS", "3")
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    client = constructor.return_value.__enter__.return_value
    client.chat.completions.create.return_value = response("pass")
    items = [
        utils.process_results({**doc, "id": i}, ["translation"])["ltb_pass_rate"]
        for i in range(20)
    ]
    assert utils.aggregate_pass_rate(items) == 1
    assert client.chat.completions.create.call_count == 40


@pytest.mark.parametrize("failure", ["api", "truncation"])
def test_aggregation_stops_after_failure_with_earlier_request_running(
    monkeypatch, doc, tmp_path, failure
):
    monkeypatch.setenv("LTB_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LTB_JUDGE_WORKERS", "2")
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    client = constructor.return_value.__enter__.return_value
    first_started = threading.Event()
    context = threading.local()
    workers = set()
    audits = []
    submitted = []
    original_score = utils._score_example

    def score(client, model, item, audit, stop=None):
        context.stop = stop
        workers.add(threading.current_thread())
        audits.append(audit)
        return original_score(client, model, item, audit, stop=stop)

    class CountingExecutor(utils.ThreadPoolExecutor):
        def submit(self, fn, item):
            submitted.append(item["id"])
            return super().submit(fn, item)

    def create(**request):
        prompt = request["messages"][0]["content"]
        if "Input: source-0\n" in prompt:
            first_started.set()
            # The first request remains in flight until the second fails.
            assert context.stop.wait(5), "Failure was not signalled to workers"
            constructor.return_value.__exit__.assert_not_called()
            assert all(not audit.handle.closed for audit in audits)
            return response("pass")
        assert "Input: source-1\n" in prompt, "Unexpected later judge request"
        assert first_started.wait(5)
        if failure == "api":
            raise RuntimeError("API unavailable")
        return response("unfinished", "length")

    monkeypatch.setattr(utils, "_score_example", score)
    monkeypatch.setattr(utils, "ThreadPoolExecutor", CountingExecutor)
    client.chat.completions.create.side_effect = create
    items = [
        utils.process_results(
            {**doc, "id": i, "source_text": f"source-{i}"}, ["translation"]
        )["ltb_pass_rate"]
        for i in range(20)
    ]
    with pytest.raises(RuntimeError, match="API unavailable|truncated"):
        utils.aggregate_pass_rate(items)
    assert submitted == [0, 1]
    assert client.chat.completions.create.call_count == 2
    assert all(not worker.is_alive() for worker in workers)
    assert all(audit.handle.closed for audit in audits)
    constructor.return_value.__exit__.assert_called_once()
    output = tmp_path / "audit"
    assert len(json.loads((output / "ltb_submission.json").read_text())) == 20
    records = [
        json.loads(line)
        for line in (output / "judge_responses.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert {record["id"] for record in records} == {0, 1}
    assert all(record["event"] == "rule" for record in records)


@pytest.mark.parametrize("failure", ["api", "truncation"])
def test_audit_survives_judge_failure(doc, tmp_path, failure):
    item = utils.process_results(doc, ["Übersetzung\n"])["ltb_pass_rate"]
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        response("pass"),
        RuntimeError("secret credential")
        if failure == "api"
        else response("unfinished", "length"),
    ]
    with utils.JudgeLog([item]) as audit:
        # Predictions must already be readable before any judge request.
        assert json.loads((audit.directory / "ltb_submission.json").read_text()) == [
            {"id": doc["id"], "translation": "Übersetzung\n"}
        ]
        with pytest.raises(RuntimeError):
            utils._score_example(client, "judge", item, audit)
    text = (tmp_path / "audit" / "judge_responses.jsonl").read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert len(records) == 2
    assert records[0]["passed"] is True
    assert (
        records[0]["response"]["choices"][0]["message"]["reasoning"]
        == "judge reasoning"
    )
    assert records[0]["request"]["messages"][0]["content"] == utils.verification_prompt(
        item["source_text"], item["translation"], item["verification_rules"][0]
    )
    assert "secret credential" not in text
    if failure == "api":
        assert records[1]["error_type"] == "RuntimeError"
    else:
        assert records[1]["truncated"] is True
        assert records[1]["passed"] is None


def test_concurrent_audit_and_no_overwrite(doc, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    items = [
        utils.process_results({**doc, "id": i}, ["translation"])["ltb_pass_rate"]
        for i in range(20)
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = response("pass")
    with utils.JudgeLog(items) as audit, ThreadPoolExecutor(max_workers=4) as executor:
        assert (
            list(
                executor.map(
                    lambda item: utils._score_example(client, "judge", item, audit),
                    items,
                )
            )
            == [1] * 20
        )
    path = tmp_path / "audit" / "judge_responses.jsonl"
    before = path.read_text()
    records = [json.loads(line) for line in before.splitlines()]
    assert len(records) == 60
    assert {r["id"] for r in records if r["event"] == "example"} == set(range(20))
    with pytest.raises(FileExistsError):
        utils.JudgeLog(items)
    assert path.read_text() == before


def test_task_yaml_loads_and_builds_generation_request(tmp_path, doc):
    from pathlib import Path

    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks._yaml_loader import load_yaml

    config = load_yaml(Path(utils.__file__).with_name("ltb_v1.yaml"), resolve_func=True)
    assert config["dataset_path"] == "zouhar/last-translation-benchmark"
    assert (
        config["dataset_kwargs"]["revision"]
        == "a483825ddbe2d7756f5bdfb1e4f611bee9026c4c"
    )
    config["dataset_path"] = "json"
    config["dataset_kwargs"] = {
        "data_files": write_data(tmp_path, [doc]),
        "cache_dir": str(tmp_path / "cache"),
    }
    task = ConfigurableTask(config=config)
    assert task.config.task == "ltb_v1"
    task.build_all_requests(limit=1, rank=0, world_size=1)
    assert len(task.instances) == 1
    assert task.instances[0].request_type == "generate_until"
    assert task.instances[0].args[0] == utils.doc_to_text(doc)
    assert task.doc_to_target(task.test_docs()[0]) == ""


def test_task_is_discoverable():
    from pathlib import Path

    from lm_eval.tasks import TaskManager

    manager = TaskManager(
        include_path=Path(utils.__file__).parent, include_defaults=False
    )
    assert "ltb_v1" in manager.all_tasks


def test_upstream_evaluator_scores_and_exports(monkeypatch, tmp_path, doc):
    from pathlib import Path

    from lm_eval.api.task import ConfigurableTask
    from lm_eval.evaluator import evaluate
    from lm_eval.models.dummy import DummyLM
    from lm_eval.tasks._yaml_loader import load_yaml

    monkeypatch.setenv("LTB_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("LTB_JUDGE_WORKERS", "1")
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    client = constructor.return_value.__enter__.return_value
    client.chat.completions.create.side_effect = [
        response("pass"),
        response("pass"),
        response("fail"),
    ]
    docs = [
        doc,
        {**doc, "id": 338, "verification_rules": ["one rule"]},
        {**doc, "id": 339},
    ]
    config = load_yaml(Path(utils.__file__).with_name("ltb_v1.yaml"), resolve_func=True)
    config["dataset_path"] = "json"
    config["dataset_kwargs"] = {
        "data_files": write_data(tmp_path, docs),
        "cache_dir": str(tmp_path / "cache"),
    }
    task = ConfigurableTask(config=config)
    model = DummyLM()
    model.generate_until = MagicMock(
        return_value=["Translation A", "Translation B", ""]
    )
    result = evaluate(
        lm=model,
        task_dict={"tasks": {"ltb_v1": task}},
        bootstrap_iters=0,
        log_samples=True,
    )
    assert result["results"]["ltb_v1"]["ltb_pass_rate,none"] == pytest.approx(1 / 3)
    assert len(result["samples"]["ltb_v1"]) == 3
    assert client.chat.completions.create.call_count == 3
    generated_requests = model.generate_until.call_args.args[0]
    assert [r.args[0] for r in generated_requests] == [
        utils.doc_to_text(d) for d in docs
    ]
    submission = json.loads((tmp_path / "audit" / "ltb_submission.json").read_text())
    assert submission == [
        {"id": 337, "translation": "Translation A"},
        {"id": 338, "translation": "Translation B"},
        {"id": 339, "translation": ""},
    ]
