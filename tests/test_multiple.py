"""Tests for MultiPL-E formatting and scoring without executing generated code."""

import json

import pytest

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.manager import TaskManager
from lm_eval.tasks.multiple import utils


SOURCE_COMMIT = "3025a531af7450e7df8b96fe0440e9804480bbad"
DATASET_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"
LANGUAGES = ("cpp", "java", "js", "php", "rs", "sh")


def test_doc_to_text_matches_reference_strip():
    assert utils.doc_to_text({"prompt": "\nfn answer() {\n"}) == "fn answer() {"


def test_build_predictions_matches_reference_program_stitching():
    doc = {
        "name": "HumanEval_0_example",
        "language": "rs",
        "prompt": "fn answer() {\n",
        "tests": "}\nfn main() { assert_eq!(answer(), 42); }",
    }

    result = utils.build_predictions([["    42"]], [doc])

    assert len(result) == 1
    payload = json.loads(result[0][0])
    assert payload == {
        "schema_version": 1,
        "name": "HumanEval_0_example",
        "language": "rs",
        "program": (
            "fn answer() {\n    42\n}\nfn main() { assert_eq!(answer(), 42); }"
        ),
    }


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [
        (20, 0, 1, 0.0),
        (20, 20, 1, 1.0),
        (20, 1, 1, pytest.approx(0.05)),
        (200, 1, 10, pytest.approx(0.05)),
        (200, 2, 100, pytest.approx(0.7512562814070352)),
    ],
)
def test_estimate_pass_at_k(n, c, k, expected):
    assert utils.estimate_pass_at_k(n, c, k) == expected


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(0, 0, 1), (10, -1, 1), (10, 11, 1), (10, 1, 0), (10, 1, 11)],
)
def test_estimate_pass_at_k_rejects_invalid_counts(n, c, k):
    with pytest.raises(ValueError):
        utils.estimate_pass_at_k(n, c, k)


def test_pass_at_k_delegates_only_to_external_executor(monkeypatch):
    requests = [
        json.dumps(
            {
                "schema_version": 1,
                "name": "one",
                "language": "js",
                "program": "function one() {}",
            }
        )
        for _ in range(4)
    ]
    seen = []

    def fake_executor(decoded):
        seen.extend(decoded)
        return [True, False, False, True]

    monkeypatch.setattr(utils, "_run_executor", fake_executor)

    result = utils.pass_at_k([], [requests], k=[1, 2])

    assert len(seen) == 4
    assert result == {"pass@1": 0.5, "pass@2": pytest.approx(5 / 6)}


def test_executor_requires_explicit_configuration(monkeypatch):
    monkeypatch.delenv(utils.EXECUTOR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=utils.EXECUTOR_ENV):
        utils._executor_command()


def test_executor_accepts_official_result_shape():
    assert utils._result_passed({"id": 0, "status": "OK", "exit_code": 0}, 0)
    assert not utils._result_passed(
        {"id": 0, "status": "RuntimeError", "exit_code": 1}, 0
    )


def test_all_requested_tasks_and_groups_are_registered():
    manager = TaskManager()
    expected_tasks = {
        f"multiple_{benchmark}_{language}{suffix}"
        for benchmark in ("humaneval", "mbpp")
        for language in LANGUAGES
        for suffix in ("", "_pass_at_10_100")
    }

    assert expected_tasks.issubset(manager.all_subtasks)
    assert {
        "multiple",
        "multiple_pass_at_1",
        "multiple_pass_at_10_100",
    }.issubset(manager.all_groups)


@pytest.mark.parametrize("profile", ["_pass_at_1.yaml", "_pass_at_10_100.yaml"])
def test_profiles_pin_source_and_dataset(profile):
    config = load_yaml(
        f"lm_eval/tasks/multiple/{profile}", resolve_func=False, recursive=False
    )

    assert config["dataset_kwargs"]["revision"] == DATASET_REVISION
    assert config["metadata"]["source_commit"] == SOURCE_COMMIT
    assert config["metadata"]["dataset_revision"] == DATASET_REVISION
    assert config["unsafe_code"] is True
    assert "no canonical translated completion" in config["bpb_unsupported_reason"]
