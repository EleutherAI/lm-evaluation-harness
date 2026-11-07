"""The CLI should update details metadata only on its sample-upload path."""

import argparse
import json
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import pytest
from huggingface_hub import DatasetCard

import lm_eval
from lm_eval._cli.run import Run
from lm_eval.loggers import evaluation_tracker
from lm_eval.loggers.evaluation_tracker import EvaluationTracker


class RecordingHub:
    def __init__(self):
        self.files = {}
        self.listed_repos = []
        self.updated_cards = []

    def create_repo(self, repo_id, **kwargs):
        self.files.setdefault(repo_id, {})

    def upload_file(self, repo_id, path_or_fileobj, path_in_repo, **kwargs):
        name = path_in_repo.replace(chr(92), "/")
        self.files[repo_id][name] = Path(path_or_fileobj).read_bytes()

    def upload_folder(self, repo_id, folder_path, path_in_repo, **kwargs):
        directory = Path(folder_path)
        for path in directory.rglob("*"):
            if path.is_file():
                name = str(
                    PurePosixPath(path_in_repo) / path.relative_to(directory).as_posix()
                )
                self.files[repo_id][name] = path.read_bytes()

    def list_repo_files(self, repo_id, **kwargs):
        self.listed_repos.append(repo_id)
        return sorted(self.files[repo_id])

    def load_dataset(self, path, data_files, split):
        address = unquote(urlparse(data_files).path).removeprefix("/datasets/")
        repo, filename = address.split("/resolve/main/", 1)
        return {"results": [json.loads(self.files[repo][filename])["results"]]}


def _sample():
    return {
        "doc_id": 0,
        "doc": {"question": "Q"},
        "target": "A",
        "arguments": [("Q", " A")],
        "resps": [[(-1, True)]],
        "filtered_resps": [(-1, True)],
        "doc_hash": "d",
        "prompt_hash": "p",
        "target_hash": "t",
    }


def _result(log_samples):
    result = {
        "results": {"arc_easy": {"acc,none": 1.0}},
        "configs": {"arc_easy": {}},
        "config": {"batch_sizes": [1]},
        "versions": {"arc_easy": 1},
        "n-shot": {"arc_easy": 0},
    }
    if log_samples:
        result["samples"] = {"arc_easy": [_sample()]}
    return result


@pytest.fixture
def hub(monkeypatch):
    hub = RecordingHub()
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setattr(EvaluationTracker, "_api", staticmethod(lambda token: hub))
    monkeypatch.setattr(evaluation_tracker, "load_dataset", hub.load_dataset)

    def publish_card(card, repo_id, **kwargs):
        hub.updated_cards.append(repo_id)

    monkeypatch.setattr(DatasetCard, "push_to_hub", publish_card)
    return hub


def _run(monkeypatch, output_path, push_results, push_samples, log_samples, separate):
    def evaluate(**kwargs):
        kwargs["evaluation_tracker"].general_config_tracker.log_experiment_args(
            "dummy", {"model": "demo"}, None, None, False
        )
        return _result(log_samples)

    # Bypass the lazy real model-evaluation import, keeping the CLI and tracker real.
    monkeypatch.setitem(lm_eval.__dict__, "simple_evaluate", evaluate)
    Run._execute(
        argparse.Namespace(
            model="dummy",
            tasks=["arc_easy"],
            output_path=str(output_path),
            log_samples=log_samples,
            hf_hub_log_args={
                "hub_results_org": "example",
                "details_repo_name": "details",
                "results_repo_name": "summary" if separate else "details",
                "push_results_to_hub": push_results,
                "push_samples_to_hub": push_samples,
            },
        )
    )


@pytest.mark.parametrize("push_results", [False, True])
@pytest.mark.parametrize("push_samples", [False, True])
@pytest.mark.parametrize("log_samples", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_hub_card_follows_sample_upload(
    monkeypatch, tmp_path, hub, push_results, push_samples, log_samples, separate
):
    _run(monkeypatch, tmp_path, push_results, push_samples, log_samples, separate)

    expected_cards = ["example/details-private"] if push_samples and log_samples else []
    assert hub.listed_repos == expected_cards
    assert hub.updated_cards == expected_cards
    if push_results:
        repo = "example/summary-private" if separate else "example/details-private"
        assert any(name.endswith(".json") for name in hub.files[repo])
    if push_samples and log_samples:
        assert any(
            name.endswith(".jsonl") for name in hub.files["example/details-private"]
        )


def test_results_only_does_not_rebuild_an_existing_details_card(
    monkeypatch, tmp_path, hub
):
    previous = EvaluationTracker(
        output_path=str(tmp_path / "previous"),
        hub_results_org="example",
        details_repo_name="details",
        push_samples_to_hub=True,
    )
    previous.general_config_tracker.log_experiment_args(
        "dummy", {"model": "previous"}, None, None, False
    )
    previous.save_results_aggregated(_result(False))
    previous.save_results_samples("arc_easy", [_sample()])
    original_details = dict(hub.files["example/details-private"])

    _run(monkeypatch, tmp_path / "current", True, False, False, True)

    assert hub.listed_repos == []
    assert hub.updated_cards == []
    assert hub.files["example/details-private"] == original_details
    assert any(name.endswith(".json") for name in hub.files["example/summary-private"])
