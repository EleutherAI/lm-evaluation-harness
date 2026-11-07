"""Dataset-card globs describe repository paths independently of the host OS."""

import os
import posixpath
from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from uuid import uuid4

import fsspec
import fsspec.utils
import pytest
from datasets.data_files import resolve_pattern
from huggingface_hub import DatasetCard

from lm_eval.loggers import evaluation_tracker


STAMP = "2026-11-08T05-00-00.000000"
RESULT = f"results_{STAMP}.json"
SAMPLE = f"samples_task_{STAMP}.jsonl"


def _make_card(monkeypatch, path_type=None, normalize_dates=True):
    tracker = evaluation_tracker.EvaluationTracker(
        hub_results_org="example", details_repo_name="results", public_repo=True
    )
    tracker.general_config_tracker.log_experiment_args(
        "dummy", {"model": "demo"}, None, None, False
    )
    tracker.api = SimpleNamespace(
        list_repo_files=lambda **kwargs: [f"demo/{RESULT}", f"demo/{SAMPLE}"]
    )
    if path_type is not None:
        monkeypatch.setattr(evaluation_tracker, "Path", path_type)
    if normalize_dates:
        # Exercise both config builders independently of filename date parsing.
        monkeypatch.setattr(evaluation_tracker, "get_file_datetime", lambda name: STAMP)
    monkeypatch.setattr(
        evaluation_tracker,
        "load_dataset",
        lambda *args, **kwargs: {"results": [{"task": {"acc,none": 1.0}}]},
    )
    cards = []
    monkeypatch.setattr(
        DatasetCard, "push_to_hub", lambda card, *args, **kwargs: cards.append(card)
    )

    tracker.recreate_metadata_card()

    assert len(cards) == 1
    # Parse the rendered YAML again, as a dataset-card consumer would.
    return DatasetCard(str(cards[0])).data.to_dict()


@pytest.mark.parametrize("path_type", [PurePosixPath, PureWindowsPath])
def test_aggregate_and_sample_card_globs_use_posix_paths(monkeypatch, path_type):
    card = _make_card(monkeypatch, path_type)
    configs = {config["config_name"]: config for config in card["configs"]}

    for name, filename in [("demo__results", RESULT), ("demo__task", SAMPLE)]:
        files = configs[name]["data_files"]
        assert {row["split"] for row in files} == {
            "latest",
            "2026_11_08T05_00_00.000000",
        }
        assert all(row["path"] == [f"**/{filename}"] for row in files)


def test_native_sample_card_path_with_real_filename_parser(monkeypatch):
    card = _make_card(monkeypatch, normalize_dates=False)
    sample = next(
        config for config in card["configs"] if config["config_name"] == "demo__task"
    )

    assert all(row["path"] == [f"**/{SAMPLE}"] for row in sample["data_files"])


def test_windows_generated_card_globs_resolve_for_posix_consumer(monkeypatch):
    card = _make_card(monkeypatch, PureWindowsPath)
    directory = f"/metadata-card-{uuid4().hex}"
    filesystem = fsspec.filesystem("memory")
    for filename in [RESULT, SAMPLE]:
        filesystem.pipe(f"{directory}/demo/{filename}", b"{}")
    # Keep the real resolver and glob implementation, using POSIX separators.
    platform_os = SimpleNamespace(**vars(os))
    platform_os.path = posixpath
    monkeypatch.setattr(fsspec.utils, "os", platform_os)

    for config in card["configs"]:
        for row in config["data_files"]:
            (pattern,) = row["path"]
            matches = resolve_pattern(pattern, f"memory://{directory}")
            assert len(matches) == 1
